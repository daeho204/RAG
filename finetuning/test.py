import json
import os

with open("dataset/training_data_claude_output_messages_type.jsonl", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]
    
for item in data[:1]:  # 첫 3개 항목만 출력
    # print(item['messages'][0].keys()) 
    for msg in item['messages']:
        print(f"Role: {msg['role']}")
        print(f"Content: {msg['content'][:100]}...")  # 내용의 앞 100자만 출력
        print("-" * 50)
    # print(json.dumps(item, ensure_ascii=False, indent=2))