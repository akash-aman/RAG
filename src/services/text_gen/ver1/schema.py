from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Any, Dict


class ChatMessage(BaseModel):
    role: Literal["agent", "user", "assistant"] = Field(
        ..., description="The role of the message sender."
    )
    content: str = Field(..., min_length=1, description="The content of the message.")


class ChatRequest(BaseModel):
    model: Literal[
        "gpt-3.5-turbo",
        "gpt-4",
    ] = Field(
        default="gpt-4",
        description="The model to use for the chat completion.",
    )
    messages: List[ChatMessage] = Field(
        ...,
        min_items=1,
        description="A list of messages representing the conversation so far.",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. Higher values make output more random.",
    )
    max_tokens: Optional[int] = Field(
        default=512,
        gt=0,
        description="The maximum number of tokens to generate in the completion.",
    )
    stream: bool = Field(default=False, description="Whether to stream the response.")

    role: Optional[Literal["assistant", "agent"]] = Field(
        default="system",
        description="Role of the message sender. If not provided, defaults to 'user'.",
    )


class ChatChoice(BaseModel):
    index: int = Field(..., ge=0, description="Index of the completion choice.")
    message: ChatMessage = Field(..., description="The generated assistant message.")
    finish_reason: Optional[str] = Field(
        default=None,
        description="The reason the generation finished (e.g., stop, length).",
    )


class ChatResponse(BaseModel):
    id: str = Field(..., description="Unique identifier for the chat completion.")
    object: str = Field(default="chat.completion")
    created: int = Field(..., description="Unix timestamp when response was created.")
    model: str = Field(..., description="The model used for generation.")
    choices: List[ChatChoice] = Field(..., description="List of message choices.")
    usage: Optional[Dict[str, int]] = Field(
        default=None, description="Token usage statistics."
    )


class ChatStreamDelta(BaseModel):
    role: Optional[str] = Field(None, description="Role (only in first chunk)")
    content: Optional[str] = Field(None, description="Partial content response.")


class ChatStreamChoice(BaseModel):
    delta: ChatStreamDelta = Field(..., description="Delta update for streaming.")
    index: int = Field(..., ge=0)
    finish_reason: Optional[str] = Field(
        default=None, description="Set when the message is done."
    )


class ChatStreamChunk(BaseModel):
    id: str = Field(..., description="Unique ID of the stream chunk.")
    object: str = Field(default="chat.completion.chunk")
    created: int = Field(..., description="Timestamp of creation.")
    model: str = Field(..., description="The model used.")
    choices: List[ChatStreamChoice] = Field(
        ..., description="Streaming response choices."
    )


class TrainingExample(BaseModel):
    input: str = Field(..., description="Input text for training")
    output: str = Field(..., description="Target output text for training")


class TrainingRequest(BaseModel):
    training_data: List[TrainingExample] = Field(
        ..., min_items=1, description="List of training examples"
    )
    epochs: int = Field(
        default=1, ge=1, le=100, description="Number of training epochs"
    )
    learning_rate: float = Field(
        default=1e-5, ge=1e-7, le=1e-2, description="Learning rate for optimization"
    )
    batch_size: int = Field(
        default=4, ge=1, le=64, description="Batch size for training"
    )
    model_name: Optional[str] = Field(
        default=None, description="Name to save the fine-tuned model as"
    )


class TrainingResponse(BaseModel):
    status: str = Field(..., description="Status of the training process")
    message: str = Field(..., description="Message describing the result")
    details: Dict[str, Any] = Field(
        ..., description="Detailed information about the training run"
    )
