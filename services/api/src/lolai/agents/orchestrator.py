import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from ollama import chat
from langchain_ollama import OllamaLLM


load_dotenv(Path(__file__).resolve().parents[4] / "local.env")

qwen = OllamaLLM(model="qwen3:8b")

messages = [
    ("system", "you are qwen"),
    ("user", "who are you"),
]

res = qwen.stream(input=messages)

for chunk in res:
    print(chunk, end="", flush=True)
# client = OpenAI(
#     # api_key=os.environ["deepseek_api_key"],
#     # base_url=os.environ["deepseek_api_key"],
#     base_url=os.environ["local_base_url"]
# )

# completion = client.chat.completions.create(
#     model=os.environ["local_model"],
#     messages=[
#         {"role": "system", "content": "You are chatgpt"},
#         {"role": "user", "content": "who are you"}
#     ],
# )

# print(completion.choices[0].message.content)