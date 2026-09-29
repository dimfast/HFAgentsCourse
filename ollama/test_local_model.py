import ollama

response = ollama.chat(
    model="qwen2.5:3b",
    messages=[
        {
            "role": "system",
            "content": "You are the useful assistant. Give me only short and detailed answers.",
        },
        {"role": "user", "content": "Hello! What is the path of your working directory?"},
    ],
)

print("Response:")
print(response["message"]["content"])