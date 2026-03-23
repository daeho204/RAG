import os
import pandas as pd
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_excel(os.path.join(BASE_DIR, "dataset", "glossary_definition_v2026.03.03.00.xlsx"))

eng_data = []
kr_data = []
for _, row in df.iterrows():
    term = str(row["영한 색인"]).strip()   # 공백 주의

    if ":" not in term:
        continue

    eng, kr = term.split(":", 1)
    eng = eng.strip()
    kr = kr.strip()
    eng_data.append(eng)
    kr_data.append(kr)
    

# res = pd.DataFrame({
#     "영어": eng_data,
#     "한국어": kr_data
# })
res = {
    "영어": eng_data,
    "한국어": kr_data
}

output_path = os.path.join(BASE_DIR, "dataset", "glossary_output_1.json")
# res.to_excel(output_path, index=False)    
# with open(output_path, "w", encoding="utf-8") as f:
#     json.dump(res, f, ensure_ascii=False, indent=2)

# preview = {k: v[:3] for k, v in res.items()}
# print(preview)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(dict(zip(eng_data, kr_data)), f, ensure_ascii=False, indent=2)