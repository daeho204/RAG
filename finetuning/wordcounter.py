import os
import json

from collections import Counter

counter = Counter()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
jlist=[]
with open("finetuning/dataset/training_data_claude.jsonl", encoding="utf-8") as f:
    for line in f:
        jlist.append(json.loads(line))

paragraphs=[]
word_list = []
for i in range(len(jlist)):
    paragraphs.append(jlist[i]['영문'])

# print(paragraphs[479:493])
with open("finetuning/dataset/glossary_output_1.json", encoding="utf-8") as f:
    word_list.append(json.load(f))     
eng_list = word_list[0].keys()


for p in paragraphs[479:493]:
    p_lower = p.lower()
    for word in eng_list:
        counter[word] += p_lower.count(word.lower())

print(counter)
