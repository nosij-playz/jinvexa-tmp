import os

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVAPI_KEY")
)

messages = [
    {
        "role": "system",
        "content": (
            "You are a helpful AI assistant. "
            "Always answer in English unless the user asks another language."
        )
    }
]

print("=" * 60)
print("NVIDIA DeepSeek Chatbot")
print("Type 'exit' or 'quit' to end.")
print("=" * 60)

while True:
    user = input("\nYou: ").strip()

    if user.lower() in {"exit", "quit"}:
        print("\nGoodbye!")
        break

    if not user:
        continue

    messages.append({
        "role": "user",
        "content": user
    })

    try:
        completion = client.chat.completions.create(
            model="deepseek-ai/deepseek-v4-flash",
            messages=messages,
            temperature=0.7,
            top_p=0.95,
            max_tokens=4096,
            extra_body={
                "chat_template_kwargs": {
                    "thinking": True,
                    "reasoning_effort": "medium"
                }
            },
            stream=False
        )

        message = completion.choices[0].message

        # Optional reasoning
        reasoning = (
            getattr(message, "reasoning", None)
            or getattr(message, "reasoning_content", None)
        )

        if reasoning:
            print("\n--- Reasoning ---")
            print(reasoning)

        answer = message.content

        print("\nAssistant:")
        print(answer)

        messages.append({
            "role": "assistant",
            "content": answer
        })

    except Exception as e:
        print(f"\nError: {e}")