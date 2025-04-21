import bentoml
import os
import time
import uuid

# from pydantic import field_validator
from ver1 import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatChoice,
    ChatStreamChunk,
    ChatStreamChoice,
    ChatStreamDelta,
    TrainingRequest,
    TrainingResponse,
)


@bentoml.service(
    name="language_model",
    traffic={"rules": {"v1": "v1", "v2": "v2"}},
)
class LanguageModelService(bentoml.Service):
    def __init__(self):
        self.model_path = os.environ.get("TT_MODEL_PATH", "default-model-path")
        self.model = None
        self.tokenizer = None

    @bentoml.api(route="/v1/chat")
    def chat_v1(self, data: ChatRequest) -> ChatResponse:
        response_text = "This is a sample assistant response."
        response_message = ChatMessage(role="assistant", content=response_text)

        choice = ChatChoice(index=0, message=response_message, finish_reason="stop")

        return ChatResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            object="chat.completion",
            created=int(time.time()),
            model=data.model,
            choices=[choice],
            usage={"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
        )

    @bentoml.api(route="/v1/stream")
    async def stream_v1(self, data: ChatRequest):
        parts = ["This ", "is ", "a ", "streamed ", "response."]
        for part in parts:
            chunk = ChatStreamChunk(
                id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
                object="chat.completion.chunk",
                created=int(time.time()),
                model=data.model,
                choices=[
                    ChatStreamChoice(
                        index=0,
                        delta=ChatStreamDelta(content=part, role="assistant"),
                        finish_reason=None,
                    )
                ],
            )
            yield f"data: {chunk.json()}\n\n"
            time.sleep(5)  # Reduced sleep time for faster response

        # End signal
        yield "data: [DONE]\n\n"

    @bentoml.api(route="/v1/train")
    def train_v1(self, data: TrainingRequest) -> TrainingResponse:
        try:
            # The field validators in TrainingRequest have already validated the input data
            # Log training parameters
            print(
                f"Starting training with parameters: epochs={data.epochs}, lr={data.learning_rate}, batch_size={data.batch_size}"
            )
            print(f"Training data size: {len(data.training_data)} examples")

            # In a real implementation, you would:
            # 1. Prepare the data for fine-tuning
            # 2. Set up the training configuration
            # 3. Run the training loop
            # 4. Save the model checkpoint

            # Simulate training process
            training_time = (
                len(data.training_data) * data.epochs * 0.1
            )  # Simulated time calculation

            # Return training results using the TrainingResponse model
            return TrainingResponse(
                status="success",
                message="Training completed successfully",
                details={
                    "model_path": self.model_path,
                    "model_name": data.model_name or f"fine-tuned-{int(time.time())}",
                    "epochs_completed": data.epochs,
                    "training_samples": len(data.training_data),
                    "training_time_seconds": training_time,
                    "timestamp": int(time.time()),
                },
            )
        except Exception as e:
            # Handle any exceptions during training
            return TrainingResponse(
                status="error",
                message=f"Training failed: {str(e)}",
                details={"error_type": type(e).__name__, "timestamp": int(time.time())},
            )
