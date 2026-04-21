import os
from dotenv import load_dotenv
from hello_agents import HelloAgentsLLM

load_dotenv()

llm = HelloAgentsLLM(
    provider="custom",
    model=os.getenv("LLM_MODEL_ID"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL")
)

try:
    response = llm.invoke(messages=[{"role": "user", "content": "你好"}])
    print("Success:", response)
except Exception as e:
    print("Error:", e)