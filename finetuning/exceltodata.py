# import os
# import pandas as pd
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# df = pd.read_excel(os.path.join(BASE_DIR, "dataset", "glossary_definition_v2026.03.03.00.xlsx"))
# # df = pd.read_excel("./dataset/glossary_definition_v2026.03.03.00.xlsx")
# print(df.columns.tolist())
# print(df.head(2))

import os
import pandas as pd
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_excel(os.path.join(BASE_DIR, "dataset", "glossary_definition_v2026.03.03.00.xlsx"))

data = []
for _, row in df.iterrows():
    term    = str(row["영한 색인"]).strip()   # 공백 주의
    kor_def = str(row["한글정의"]).strip()
    eng_def = str(row["영어정의"]).strip()

    if ":" not in term:
        continue

    eng, kor = term.split(":", 1)
    eng = eng.strip()
    kor = kor.strip()

    # 형태 1: 단순 번역
    data.append({
        "prompt": f"What is the Korean translation of '{eng}'?",
        "response": kor
    })

    # 형태 2: 한글 정의 포함
    if kor_def and kor_def != "nan":
        data.append({
            "prompt": f"'{eng}'의 한국어 번역과 정의를 알려주세요.",
            "response": f"{kor}\n\n{kor_def}"
        })

    # 형태 3: 영어 정의 포함
    if eng_def and eng_def != "nan":
        data.append({
            "prompt": f"Define the term '{eng}'.",
            "response": f"{kor} — {eng_def}"
        })

output_path = os.path.join(BASE_DIR, "dataset", "glossary_train.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"총 {len(data)}개 학습 데이터 생성 완료")
print(f"저장 위치: {output_path}")