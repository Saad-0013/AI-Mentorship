import time
from groq import Groq
from models.schemas import Message

# A typed error specifically for the route handler to catch
class GroqServiceError(Exception):
    pass

def generate_reply(messages: list[Message], system_prompt: str, api_key: str) -> tuple[str, int, int, int]:
    client = Groq(api_key=api_key)
    
    # Prepend the system instructions to the chat history
    api_messages = [{"role": "system", "content": system_prompt}]
    for msg in messages:
        api_messages.append({"role": msg.role, "content": msg.content})

    start_time = time.perf_counter()
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=api_messages,
            temperature=0.0
        )
    except Exception as e:
        # Wrap the failure in our custom typed error
        raise GroqServiceError(f"Groq API call failed: {str(e)}") from e

    # Calculate latency and extract the required token counts
    latency_ms = int((time.perf_counter() - start_time) * 1000)
    reply = response.choices[0].message.content
    tokens_in = response.usage.prompt_tokens
    tokens_out = response.usage.completion_tokens

    return reply, latency_ms, tokens_in, tokens_out
