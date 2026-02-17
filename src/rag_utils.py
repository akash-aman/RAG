import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import time

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
import openai

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def load_pdf(self, file_path: str) -> str:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

    def chunk_text(self, text: str) -> List[str]:
        return self.text_splitter.split_text(text)

class EmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

class VectorStore:
    def __init__(self, collection_name: str = "rag_collection", dim: int = 384):
        self.collection_name = collection_name
        self.dim = dim
        self.connect_milvus()
        self.collection = self._get_or_create_collection()

    def connect_milvus(self):
        try:
            connections.connect("default", host="localhost", port="19530")
        except Exception as e:
            print(f"Failed to connect to Milvus: {e}")
            # Retry logic or handle gracefully? For now, we assume it works or fails hard.

    def _get_or_create_collection(self) -> Collection:
        if utility.has_collection(self.collection_name):
            return Collection(self.collection_name)
        
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
        ]
        schema = CollectionSchema(fields, "RAG Knowledge Base")
        collection = Collection(self.collection_name, schema)
        
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        return collection

    def insert_embeddings(self, texts: List[str], embeddings: List[List[float]], source: str):
        data = [
            texts,
            embeddings,
            [source] * len(texts)
        ]
        self.collection.insert(data)
        self.collection.flush()

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        self.collection.load()
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["text", "source"]
        )
        
        retrieved = []
        for hits in results:
            for hit in hits:
                retrieved.append({
                    "id": hit.id,
                    "distance": hit.distance,
                    "text": hit.entity.get("text"),
                    "source": hit.entity.get("source")
                })
        return retrieved

class LLMClient:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        if base_url:
            self.client = openai.OpenAI(base_url=base_url, api_key=api_key or "lm-studio")
            self.model = "local-model" # LM Studio usually ignores this or uses the loaded model
        elif api_key:
            self.client = openai.OpenAI(api_key=api_key)
            self.model = "gpt-3.5-turbo"
        else:
            self.client = None

    def generate_response(self, query: str, context: List[Dict[str, Any]]) -> str:
        if not self.client:
            return "No LLM client configured. Set OPENAI_API_KEY or provide a local base_url."
        
        context_text = "\n\n".join([f"Source: {item['source']}\nContent: {item['text']}" for item in context])
        
        prompt = f"""You are a helpful assistant. Use the following context to answer the user's question.
If the answer is not in the context, say you don't know.

Context:
{context_text}

Question:
{query}
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {e}"
