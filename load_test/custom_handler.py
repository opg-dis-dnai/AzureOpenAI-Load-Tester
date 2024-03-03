import httpx
from tiktoken import Encoding
from openai import APIError


class CustomClient:
    # Replace and define your own custom client class here
    def __init__(
        self, endpoint: str, api_key: str, tiktoken_encoding: Encoding, **kwargs
    ) -> None:
        self.endpoint = endpoint
        self.api_key = api_key
        self.tiktoken = tiktoken_encoding
        self.params = kwargs

    async def custom_request_handler(
        self, model: str, message: str, max_tokens: int = None
    ) -> httpx.Response:
        # Replace with the actual request logic for your custom API
        payload = {"messages": [{"role": "user", "content": message}], "model": model}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.endpoint,
                json=payload,
                headers=headers,
            )

            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as http_err:

                raise APIError(f"HTTP error occurred: {http_err}") from http_err
            except httpx.RequestError as req_err:

                raise APIError(f"Request error occurred: {req_err}") from req_err

        return response

    def custom_response_handler(self, response: httpx.Response) -> int:
        # Replace with the actual response processing logic for the custom API. must return token count

        result = response.json()
        output = result["output"]
        return len(self.tiktoken.encode(output))
