from llm_client import VllmChatClient, VllmConfig
from rag.log_store import JsonlChatStore

store = JsonlChatStore("logs/chat_history.jsonl")
llm = VllmChatClient(VllmConfig(), store=store)

print(llm.chat("RAG에서 retrieval 결과를 요약해줘"))