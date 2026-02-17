"""
Multi-provider LLM client (OpenAI API-compatible).
Works with LM Studio, OpenAI, Ollama, and any OpenAI-compatible endpoint.
"""

from typing import AsyncGenerator, Optional

import httpx
import openai

from src.core.config import get_settings
from src.utils.logger.logger import get_logger

logger = get_logger(name="LLMClient")


class LLMClient:
    """Synchronous + streaming LLM client using OpenAI-compatible API."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        settings = get_settings()
        self.base_url = base_url or settings.llm_base_url
        self.api_key = api_key or settings.llm_api_key
        self.model = model or settings.llm_model
        self.default_temperature = settings.llm_temperature
        self.default_max_tokens = settings.llm_max_tokens

        self.client = openai.OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )
        logger.info(
            f"LLM client init: provider={settings.llm_provider}, "
            f"base_url={self.base_url}, model={self.model}"
        )

    def generate(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate a completion synchronously."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature or self.default_temperature,
                max_tokens=max_tokens or self.default_max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return f"[LLM Error: {e}]"

    def generate_stream(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        """Generate a streaming completion. Yields content chunks."""
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature or self.default_temperature,
                max_tokens=max_tokens or self.default_max_tokens,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"LLM streaming error: {e}")
            yield f"[LLM Error: {e}]"

    def generate_with_context(
        self,
        query: str,
        context_chunks: list[dict],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate a RAG response given query + retrieved context chunks."""
        context_text = "\n\n---\n\n".join(
            f"[Source: {c.get('source', 'unknown')}]\n{c.get('text', '')}"
            for c in context_chunks
        )

        prompt = f"""Use the following context to answer the user's question.
If the answer is not contained in the context, say you don't know based on the available information.
Always cite which source(s) you used.

Context:
{context_text}

Question: {query}

Answer:"""

        sys_prompt = system_prompt or (
            "You are a knowledgeable assistant that answers questions based on "
            "the provided context. Be concise, accurate, and cite your sources."
        )

        return self.generate(
            prompt=prompt,
            system_prompt=sys_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def generate_with_context_stream(
        self,
        query: str,
        context_chunks: list[dict],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        """Streaming version of generate_with_context."""
        context_text = "\n\n---\n\n".join(
            f"[Source: {c.get('source', 'unknown')}]\n{c.get('text', '')}"
            for c in context_chunks
        )

        prompt = f"""Use the following context to answer the user's question.
If the answer is not contained in the context, say you don't know based on the available information.
Always cite which source(s) you used.

Context:
{context_text}

Question: {query}

Answer:"""

        sys_prompt = system_prompt or (
            "You are a knowledgeable assistant that answers questions based on "
            "the provided context. Be concise, accurate, and cite your sources."
        )

        yield from self.generate_stream(
            prompt=prompt,
            system_prompt=sys_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )


# ── Singleton ────────────────────────────────────────────────────────
_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
