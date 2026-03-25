import os
import json
import random
from openai import OpenAI
from dotenv import load_dotenv  
load_dotenv()  # .env 파일에서 환경 변수 로드
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "dataset", "glossary_output_1.json"), "r", encoding="utf-8") as f:
    glossary = json.load(f)


def cluster_glossary(glossary: dict, batch_size: int = 100) -> dict:
    items = list(glossary.items())
    all_clusters = {}
    
    # batch_size씩 나눠서 클러스터링
    for i in range(0, len(items), batch_size):
        batch = dict(items[i:i+batch_size])
        word_list = "\n".join([f"{eng}: {kr}" for eng, kr in batch.items()])
        
        print(f"Batching ({i+1}~{min(i+batch_size, len(items))}/{len(items)})")
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "기술 용어 전문가입니다."},
                {"role": "user", "content": f"""
                아래 용어들을 주제별로 묶어서 JSON으로 반환해줘.
                모든 용어가 단 하나도 빠짐없이 반드시 포함되어야 해. 총 {len(batch)}개.

                [용어 목록]
                {word_list}

                [출력 형식] JSON만 반환, 다른 텍스트 없이
                {{
                "클러스터명": {{"영어용어": "한국어용어", ...}},
                ...
                }}
                """}
            ],
            response_format={"type": "json_object"}
        )
        
        batch_clusters = json.loads(response.choices[0].message.content)
        
        # 같은 클러스터명이 있으면 합치기
        for cluster_name, words in batch_clusters.items():
            if cluster_name in all_clusters:
                all_clusters[cluster_name].update(words)
            else:
                all_clusters[cluster_name] = words
    
    return all_clusters


def generate_paragraph_pair(sampled: dict, L: int = 3) -> list:
    word_list = "\n".join([f"* {eng} → {kr}" for eng, kr in sampled.items()])
    
    prompt = f"""번역 파인튜닝용 학습 데이터를 만들려고 합니다.
아래 용어들을 사용해서 4~5개의 단어가 포함되고 5~6줄 이상으로 이뤄진 한/영 문단 쌍을 {L}개 생성해주세요.

[선택된 용어]
{word_list}

[조건]
1. 영문 문단에는 위 영어 용어를 모두 정확히 1회 이상 포함
2. 한글 문단에는 대응되는 한국어 용어를 모두 정확히 1회 이상 포함
3. 한/영 문단은 의미적으로 대응되는 번역 관계여야 함
4. 용어 표기는 위에 제시된 표기 그대로 사용
5. 자연스러운 기술 문서 형태로 작성
6. {L}개의 문단 쌍은 서로 다른 내용이어야 함

[출력 형식] JSON만 반환, 다른 텍스트 없이
[
  {{
    "영문": "...",
    "한글": "..."
  }}
]

당신은 한국어-영어 기술 번역을 전문으로 하는 번역가입니다.

당신의 역할은 파인튜닝(Fine-Tuning)에 사용할 고품질 번역 데이터셋을 생성하는 것입니다.

요구사항:
- 원문의 의미를 정확하게 유지하면서 자연스럽게 번역하세요.
- 제공된 용어집(Glossary)을 반드시 준수하세요.
- 기술 문서에 적합한 정중하고 일관된 문체를 유지하세요.
- 동일한 용어는 항상 동일하게 번역하세요.
- 직역보다는 의미 전달과 정확성을 우선하세요.

출력은 반드시 모델 학습에 적합한 데이터 형식이어야 합니다.
[RULE]

1. 용어집 준수 (필수)
- source 문장에 용어집에 포함된 단어가 등장하면 반드시 지정된 번역을 그대로 사용해야 합니다.
- 용어를 변형하거나 유의어로 바꾸지 마세요.
- 용어를 누락하지 마세요.

2. 일관성 유지
- 동일한 용어는 하나의 샘플 내에서 항상 동일하게 번역해야 합니다.
- 문장마다 다른 표현으로 바꾸지 마세요.

3. 문장 자연스러움
- 한국어 번역은 자연스럽고 기술 문서에 적합한 형태로 작성하세요.
- 의미를 훼손하는 직역은 피하세요.

4. 길이 제한
- 다음 중 하나를 생성하세요:
  (a) 2~3줄의 짧은 문장
  (b) 5~6줄 이상의 문단

5. 용어 밀도 제한
- 하나의 샘플에 너무 많은 용어를 억지로 포함시키지 마세요.
- 짧은 샘플은 1~3개 용어만 포함하세요.

6. 환각 금지
- 원문에 없는 내용을 추가하지 마세요.
- 임의로 기술 용어를 만들어내지 마세요.

7. 출력 형식 (엄격)
- 반드시 지정된 JSON 형식으로만 출력하세요.
- 추가 설명, 주석, 문장은 절대 출력하지 마세요.

[출력 형식] JSON만 반환, 다른 텍스트 없이
[
  {{
    "영문": "...",
    "한글": "..."
  }}
]
"""


    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "당신은 전문 기술 번역가입니다."},
            {"role": "user", "content": prompt}
        ]
    )
    text = response.choices[0].message.content
    text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text)


def run_pipeline(glossary: dict, M: int = 4, L: int = 3, total: int = 100):
    """
    Parameters
    ----------
    glossary : dict{영어용어: 한국어용어} 형태의 용어집 딕셔너리
    M : 한 묶음당 랜덤 추출할 용어 수 
    L : 한 묶음당 생성할 한/영 문단 쌍 수
    total : 생성할 총 묶음 수
    """
    results = []
    cluster_path = os.path.join(BASE_DIR, "dataset", "clusters.json")

    # 클러스터 파일 있으면 불러오고, 없으면 새로 생성
    # 클러스터가 있어야 토큰 소모량이 줄어든다.
    
    if os.path.exists(cluster_path):
        print("load exsiting clusters")
        with open(cluster_path, "r", encoding="utf-8") as f:
            clusters = json.load(f)
        print(f"{len(clusters)}clusters loaded")
    else:

        clusters = cluster_glossary(glossary)
        with open(cluster_path, "w", encoding="utf-8") as f:
            json.dump(clusters, f, ensure_ascii=False, indent=2)
        print(f"{len(clusters)}clusters saved")

    # 클러스터링 결과 검증
    clustered_count = sum(len(v) for v in clusters.values())
    print(f"Original Count: {len(glossary)} / Clustered Result: {clustered_count}")
    if clustered_count != len(glossary):
        print(f"{len(glossary) - clustered_count} missing")

    for cluster_name, cluster_words in clusters.items():
        if len(results) >= total:
            break
        
        items = list(cluster_words.items())
        
        while len(items) >= M:
            if len(results) >= total:  # 목표 달성하면 중단
                break
                
            sampled = dict(random.sample(items, M))
            
            try:
                pairs = generate_paragraph_pair(sampled, L)
                results.append({
                    "cluster": cluster_name,
                    "sampled_terms": sampled,
                    "paragraph_pairs": pairs
                })
                print(f"Progress: {len(results)}/{total}")
            except Exception as e:
                print(f"Error occurred ({cluster_name}): {e}")
            
            for key in sampled:
                items = [item for item in items if item[0] != key]

    return results


if __name__ == "__main__":
    M = 4
    L = 6
    TOTAL = 100
    results = run_pipeline(glossary, M=M, L=L, total=TOTAL)

    output_path = os.path.join(BASE_DIR, "dataset", "training_data_2.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"{len(results)} Created")