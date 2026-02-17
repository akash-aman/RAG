import numpy as np
from typing import List, Dict, Optional, Any
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize
import re
import hashlib

class TextTransformer:
    """
    A class for transforming text into vector embeddings with advanced chunking strategies
    optimized for RAG (Retrieval-Augmented Generation) applications.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", 
                 chunk_size: int = 512,
                 chunk_overlap: int = 128,
                 chunk_strategy: str = "sliding_window"):
        """
        Initialize the TextTransformer with specified embedding model and chunking parameters.
        
        Args:
            model_name (str): Name of the sentence-transformer model to use
            chunk_size (int): Maximum size of each text chunk in tokens
            chunk_overlap (int): Number of tokens to overlap between chunks
            chunk_strategy (str): Chunking strategy ('sliding_window', 'sentence_aware', 
                                  'paragraph', 'semantic_break', or 'hybrid')
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunk_strategy = chunk_strategy
        
        # Download NLTK resources if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')


    def _generate_chunk_id(self, text: str, index: int) -> str:
        """Generate a unique ID for a chunk based on content and position."""
        content_hash = hashlib.md5(text.encode()).hexdigest()[:10]
        return f"chunk_{index}_{content_hash}"


    def _tokenize_text(self, text: str) -> List[str]:
        """Simple tokenization by splitting on whitespace."""
        return text.split()



    def _estimate_token_count(self, text: str) -> int:
        """
        Estimate the number of tokens in a text segment.
        This is a simple approximation; for production use, 
        consider using the specific tokenizer of your embedding model.
        """
        return len(self._tokenize_text(text))



    def _get_sliding_window_chunks(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text using a sliding window approach with fixed chunk size and overlap.
        
        Returns:
            List of dictionaries containing chunk text and metadata
        """
        tokens = self._tokenize_text(text)
        chunks = []
        
        # Process text in sliding window chunks
        i = 0
        chunk_index = 0
        while i < len(tokens):
            # Get chunk tokens
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_tokens)
            
            chunks.append({
                "id": self._generate_chunk_id(chunk_text, chunk_index),
                "text": chunk_text,
                "index": chunk_index,
                "token_count": len(chunk_tokens),
                "char_start": text.find(chunk_tokens[0]),
                "char_end": text.find(chunk_tokens[-1]) + len(chunk_tokens[-1]) if chunk_tokens else 0
            })
            
            # Move to next chunk position considering overlap
            i += self.chunk_size - self.chunk_overlap
            chunk_index += 1
            
            # Avoid getting stuck in infinite loops with very small texts
            if i >= len(tokens) or (i + self.chunk_size - self.chunk_overlap) <= 0:
                break
        
        return chunks



    def _get_sentence_aware_chunks(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text into chunks respecting sentence boundaries.
        
        Returns:
            List of dictionaries containing chunk text and metadata
        """
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_token_count = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence_token_count = self._estimate_token_count(sentence)
            
            # If adding this sentence would exceed chunk size and we already have content,
            # store the current chunk and start a new one
            if current_token_count + sentence_token_count > self.chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    "id": self._generate_chunk_id(chunk_text, chunk_index),
                    "text": chunk_text,
                    "index": chunk_index,
                    "token_count": current_token_count,
                    "sentence_count": len(current_chunk)
                })
                
                # For overlap, keep some sentences from previous chunk
                overlap_tokens = 0
                overlap_sentences = []
                
                # Work backwards through current chunk to find sentences for overlap
                for s in reversed(current_chunk):
                    s_tokens = self._estimate_token_count(s)
                    if overlap_tokens + s_tokens <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_tokens += s_tokens
                    else:
                        break
                
                # Start new chunk with overlap sentences
                current_chunk = overlap_sentences
                current_token_count = overlap_tokens
                chunk_index += 1
            
            # Add current sentence to chunk
            current_chunk.append(sentence)
            current_token_count += sentence_token_count
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                "id": self._generate_chunk_id(chunk_text, chunk_index),
                "text": chunk_text,
                "index": chunk_index,
                "token_count": current_token_count,
                "sentence_count": len(current_chunk)
            })
        
        return chunks
    
    def _get_paragraph_chunks(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text by paragraphs, then combine paragraphs to respect chunk size.
        
        Returns:
            List of dictionaries containing chunk text and metadata
        """
        # Split by double newline to identify paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        current_chunk = []
        current_token_count = 0
        chunk_index = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            para_token_count = self._estimate_token_count(para)
            
            # If this paragraph alone exceeds chunk size, use sentence-aware chunking for it
            if para_token_count > self.chunk_size:
                # First save any accumulated paragraphs
                if current_chunk:
                    chunk_text = "\n\n".join(current_chunk)
                    chunks.append({
                        "id": self._generate_chunk_id(chunk_text, chunk_index),
                        "text": chunk_text,
                        "index": chunk_index,
                        "token_count": current_token_count,
                        "paragraph_count": len(current_chunk)
                    })
                    chunk_index += 1
                    current_chunk = []
                    current_token_count = 0
                
                # Now process the large paragraph with sentence chunking
                para_chunks = self._get_sentence_aware_chunks(para)
                for idx, chunk in enumerate(para_chunks):
                    chunk["id"] = self._generate_chunk_id(chunk["text"], chunk_index + idx)
                    chunk["index"] = chunk_index + idx
                
                chunks.extend(para_chunks)
                chunk_index += len(para_chunks)
                
            # If adding this paragraph would exceed chunk size, save current chunk first
            elif current_token_count + para_token_count > self.chunk_size and current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                chunks.append({
                    "id": self._generate_chunk_id(chunk_text, chunk_index),
                    "text": chunk_text,
                    "index": chunk_index,
                    "token_count": current_token_count,
                    "paragraph_count": len(current_chunk)
                })
                chunk_index += 1
                current_chunk = [para]
                current_token_count = para_token_count
                
            # Otherwise add paragraph to current chunk
            else:
                current_chunk.append(para)
                current_token_count += para_token_count
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunks.append({
                "id": self._generate_chunk_id(chunk_text, chunk_index),
                "text": chunk_text,
                "index": chunk_index,
                "token_count": current_token_count,
                "paragraph_count": len(current_chunk)
            })
        
        return chunks

    def _get_semantic_break_chunks(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text at semantic boundaries like headers, section breaks, etc.
        
        Returns:
            List of dictionaries containing chunk text and metadata
        """
        # Look for common section header patterns
        header_patterns = [
            r'^#+\s+.+$',               # Markdown headers
            r'^.+\n[=\-]{2,}$',         # Underlined headers
            r'^\d+\.\s+.+$',            # Numbered sections
            r'^[A-Z\s]+:',              # ALL CAPS followed by colon
            r'^.*?:\s*$'                # Any text ending with colon
        ]
        
        combined_pattern = '|'.join(f'({p})'for p in header_patterns)
        
        # Split text into lines
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        current_token_count = 0
        chunk_index = 0
        
        # Process each line
        for line in lines:
            line_token_count = self._estimate_token_count(line)
            
            # Check if line matches header pattern
            if re.match(combined_pattern, line, re.MULTILINE) and current_chunk:
                # Save current chunk before header
                chunk_text = '\n'.join(current_chunk)
                chunks.append({
                    "id": self._generate_chunk_id(chunk_text, chunk_index),
                    "text": chunk_text,
                    "index": chunk_index,
                    "token_count": current_token_count,
                    "line_count": len(current_chunk)
                })
                chunk_index += 1
                current_chunk = [line]
                current_token_count = line_token_count
            
            # If adding this line would exceed chunk size, save current chunk first
            elif current_token_count + line_token_count > self.chunk_size and current_chunk:
                chunk_text = '\n'.join(current_chunk)
                chunks.append({
                    "id": self._generate_chunk_id(chunk_text, chunk_index),
                    "text": chunk_text,
                    "index": chunk_index,
                    "token_count": current_token_count,
                    "line_count": len(current_chunk)
                })
                chunk_index += 1
                current_chunk = [line]
                current_token_count = line_token_count
            
            # Otherwise add line to current chunk
            else:
                current_chunk.append(line)
                current_token_count += line_token_count
                
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            chunks.append({
                "id": self._generate_chunk_id(chunk_text, chunk_index),
                "text": chunk_text,
                "index": chunk_index,
                "token_count": current_token_count,
                "line_count": len(current_chunk)
            })
        
        return chunks
    
    def _get_hybrid_chunks(self, text: str) -> List[Dict[str, Any]]:
        """
        Combine semantic and sentence-aware chunking strategies.
        First split by semantic breaks, then apply sentence-aware chunking to large sections.
        
        Returns:
            List of dictionaries containing chunk text and metadata
        """
        # First split by semantic chunks
        semantic_chunks = self._get_semantic_break_chunks(text)
        final_chunks = []
        chunk_index = 0
        
        # Process each semantic chunk
        for s_chunk in semantic_chunks:
            chunk_text = s_chunk["text"]
            
            # If chunk is already small enough, keep it as is
            if s_chunk["token_count"] <= self.chunk_size:
                s_chunk["id"] = self._generate_chunk_id(chunk_text, chunk_index)
                s_chunk["index"] = chunk_index
                final_chunks.append(s_chunk)
                chunk_index += 1
            else:
                # Further split large semantic chunks using sentence-aware chunking
                sent_chunks = self._get_sentence_aware_chunks(chunk_text)
                for sent_chunk in sent_chunks:
                    sent_chunk["id"] = self._generate_chunk_id(sent_chunk["text"], chunk_index)
                    sent_chunk["index"] = chunk_index
                    final_chunks.append(sent_chunk)
                    chunk_index += 1
                    
        return final_chunks
    
    def chunk_text(self, text: str, strategy: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Split text into chunks using the specified strategy.
        
        Args:
            text (str): The input text to chunk
            strategy (str, optional): Chunking strategy to use, overrides the default if provided
            
        Returns:
            List of dictionaries containing chunk text and metadata
        """
        strategy = strategy or self.chunk_strategy
        
        if strategy == "sliding_window":
            return self._get_sliding_window_chunks(text)
        elif strategy == "sentence_aware":
            return self._get_sentence_aware_chunks(text)
        elif strategy == "paragraph":
            return self._get_paragraph_chunks(text)
        elif strategy == "semantic_break":
            return self._get_semantic_break_chunks(text)
        elif strategy == "hybrid":
            return self._get_hybrid_chunks(text)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embeddings for a single text segment.
        
        Args:
            text (str): Text to embed
            
        Returns:
            numpy.ndarray: Vector embedding of the text
        """
        return self.model.encode(text, show_progress_bar=False)
    
    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Generate embeddings for a list of text chunks.
        
        Args:
            chunks (List[Dict]): List of chunk dictionaries with 'text' keys
            
        Returns:
            Dict: Dictionary mapping chunk IDs to chunks with embeddings added
        """
        chunk_dict = {}
        
        # Extract text from chunks for batch embedding
        texts = [chunk["text"] for chunk in chunks]
        
        # Generate embeddings in batch
        embeddings = self.model.encode(texts, show_progress_bar=False)
        
        # Add embeddings back to chunks
        for i, chunk in enumerate(chunks):
            chunk_with_embedding = chunk.copy()
            chunk_with_embedding["embedding"] = embeddings[i]
            chunk_dict[chunk["id"]] = chunk_with_embedding
            
        return chunk_dict
    
    def process_document(self, document: str, strategy: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a full document: chunk it and generate embeddings for each chunk.
        
        Args:
            document (str): The document text to process
            strategy (str, optional): Chunking strategy to use
            
        Returns:
            Dict: Dictionary containing document info, chunking strategy, and chunk data
        """
        strategy = strategy or self.chunk_strategy
        chunks = self.chunk_text(document, strategy)
        embedded_chunks = self.embed_chunks(chunks)
        
        return {
            "model_name": self.model_name,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "strategy": strategy,
            "chunks": embedded_chunks,
            "chunk_count": len(chunks),
            "embedding_dim": next(iter(embedded_chunks.values()))["embedding"].shape[0] if embedded_chunks else None
        }

    def search_similar(self, 
                     query: str, 
                     chunks: Dict[str, Dict[str, Any]], 
                     top_k: int = 3, 
                     similarity_threshold: float = 0.6) -> List[Dict[str, Any]]:
        """
        Search for chunks most similar to a query string.
        
        Args:
            query (str): The query text to search for
            chunks (Dict): Dictionary of chunks with embeddings
            top_k (int): Number of results to return
            similarity_threshold (float): Minimum similarity score (0-1)
            
        Returns:
            List[Dict]: List of matching chunks with similarity scores
        """
        # Embed the query
        query_embedding = self.embed_text(query)
        
        results = []
        
        # Calculate cosine similarity with each chunk
        for chunk_id, chunk_data in chunks.items():
            chunk_embedding = chunk_data["embedding"]
            
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, chunk_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
            )
            
            if similarity >= similarity_threshold:
                result = {
                    "chunk_id": chunk_id,
                    "text": chunk_data["text"],
                    "similarity": float(similarity),
                    "index": chunk_data["index"]
                }
                results.append(result)
        
        # Sort by similarity (highest first) and return top_k
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]