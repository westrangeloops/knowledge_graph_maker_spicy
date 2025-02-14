import os
from groq import Groq, GroqError
from ..types import LLMClient


class GroqClient(LLMClient):
    def __init__(
        self,
        model: str = "mixtral-8x7b-32768",
        temperature: float = 0.2,
        top_p: float = 1.0,
    ):
        self._model = model
        self._temperature = temperature
        self._top_p = top_p

        try:
            self._client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        except GroqError as e:
            print(f"Groq Client Initialization Error: {e}")
            self._client = None

    def generate(self, user_message: str, system_message: str) -> str:
        if not self._client:
            raise ValueError(
                "Groq client is not initialized. Please check your API key."
            )

        try:
            response = self._client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                model=self._model,
                temperature=self._temperature,
                top_p=self._top_p,
            )
            return response.choices[0].message.content
        except GroqError as e:
            print(f"Error during chat completion: {e}")
            return ""
