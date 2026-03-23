# evaluate_compare_models.py
# -*- coding: utf-8 -*-

import json
import os
import re
import string
from collections import Counter, defaultdict
from statistics import mean
from typing import Dict, Any, List, Optional

INPUT_JSONL = "out_details_remake_3.jsonl"
EVAL_JSONL = "eval_dataset.jsonl"
OUT_REPORT_JSON = "eval_report.json"


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_text(text: str) -> str:
    if text is None:
        return ""

    text = text.strip().lower()

    # unify whitespace
    text = re.sub(r"\s+", " ", text)

    # remove simple punctuation except %, ., numbers, Korean chars
    text = text.replace("％", "%")
    text = text.replace("분 이내", "분이내")
    text = text.replace("60 분", "60분")
    text = text.replace("120 분", "120분")

    # normalize parentheses spacing
    text = re.sub(r"\s*[\(\)\[\]\{\}]\s*", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_for_f1(text: str) -> List[str]:
    text = normalize_text(text)
    # simple token split; for Korean domain tasks, this is a baseline
    return text.split()


def exact_match(pred: str, refs: List[str]) -> float:
    pred_n = normalize_text(pred)
    ref_ns = [normalize_text(r) for r in refs]
    return 1.0 if pred_n in ref_ns else 0.0


def token_f1_single(pred: str, ref: str) -> float:
    pred_tokens = tokenize_for_f1(pred)
    ref_tokens = tokenize_for_f1(ref)

    if len(pred_tokens) == 0 and len(ref_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def token_f1(pred: str, refs: List[str]) -> float:
    return max(token_f1_single(pred, ref) for ref in refs)


def map_to_label(pred: str, label_space: List[str]) -> str:
    """
    Force free-form model output into one of the allowed labels.
    This mapping is important for accuracy/f1 stability.
    """
    pred_n = normalize_text(pred)

    # exact match first
    for label in label_space:
        if pred_n == normalize_text(label):
            return label

    # heuristic mapping
    if "불확실" in label_space:
        uncertain_patterns = ["모르", "불확실", "명확하지", "판단 어려", "근거 부족", "unknown"]
        if any(p in pred_n for p in uncertain_patterns):
            return "불확실"

    if "아니오" in label_space:
        no_patterns = ["아니", "없다", "없음", "유의하지 않", "충분하지 않", "not", "no"]
        if any(p in pred_n for p in no_patterns):
            return "아니오"

    if "예" in label_space:
        yes_patterns = ["예", "그렇", "맞", "있다", "유의하", "충분하", "yes"]
        if any(p in pred_n for p in yes_patterns):
            return "예"

    # fallback
    return pred.strip()


def compute_classification_metrics(
    golds: List[str],
    preds: List[str],
    labels: List[str]
) -> Dict[str, Any]:
    assert len(golds) == len(preds)

    total = len(golds)
    correct = sum(1 for g, p in zip(golds, preds) if g == p)
    accuracy = correct / total if total else 0.0

    # macro F1
    per_label = {}
    for label in labels:
        tp = sum(1 for g, p in zip(golds, preds) if g == label and p == label)
        fp = sum(1 for g, p in zip(golds, preds) if g != label and p == label)
        fn = sum(1 for g, p in zip(golds, preds) if g == label and p != label)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        per_label[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(1 for g in golds if g == label),
        }

    macro_f1 = mean(v["f1"] for v in per_label.values()) if per_label else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_label": per_label,
    }


def main():
    preds = load_jsonl(INPUT_JSONL)
    eval_rows = load_jsonl(EVAL_JSONL)

    pred_map = {row["question"]: row for row in preds}
    eval_map = {row["question"]: row for row in eval_rows}

    base_short_em = []
    base_short_f1 = []
    peft_short_em = []
    peft_short_f1 = []

    cls_labels_union = set()
    base_cls_gold = []
    base_cls_pred = []
    peft_cls_gold = []
    peft_cls_pred = []

    details = []

    for question, gold in eval_map.items():
        if question not in pred_map:
            continue

        pred_row = pred_map[question]
        task_type = gold["task_type"]

        base_out = pred_row.get("base_output", "")
        peft_out = pred_row.get("peft_output", "")

        row_detail: Dict[str, Any] = {
            "question": question,
            "task_type": task_type,
            "base_output": base_out,
            "peft_output": peft_out,
        }

        if task_type == "short_qa":
            refs = gold["reference_answers"]

            b_em = exact_match(base_out, refs)
            b_f1 = token_f1(base_out, refs)

            p_em = exact_match(peft_out, refs)
            p_f1 = token_f1(peft_out, refs)

            base_short_em.append(b_em)
            base_short_f1.append(b_f1)
            peft_short_em.append(p_em)
            peft_short_f1.append(p_f1)

            row_detail["reference_answers"] = refs
            row_detail["base_scores"] = {"em": round(b_em, 4), "f1": round(b_f1, 4)}
            row_detail["peft_scores"] = {"em": round(p_em, 4), "f1": round(p_f1, 4)}

        elif task_type == "classification":
            label_space = gold["label_space"]
            ref_label = gold["reference_label"]

            base_label = map_to_label(base_out, label_space)
            peft_label = map_to_label(peft_out, label_space)

            cls_labels_union.update(label_space)

            base_cls_gold.append(ref_label)
            peft_cls_gold.append(ref_label)
            base_cls_pred.append(base_label)
            peft_cls_pred.append(peft_label)

            row_detail["reference_label"] = ref_label
            row_detail["base_pred_label"] = base_label
            row_detail["peft_pred_label"] = peft_label

        else:
            row_detail["warning"] = f"Unsupported task_type: {task_type}"

        details.append(row_detail)

    report = {
        "short_qa": {
            "count": len(base_short_em),
            "base": {
                "exact_match": round(mean(base_short_em), 4) if base_short_em else 0.0,
                "token_f1": round(mean(base_short_f1), 4) if base_short_f1 else 0.0,
            },
            "peft": {
                "exact_match": round(mean(peft_short_em), 4) if peft_short_em else 0.0,
                "token_f1": round(mean(peft_short_f1), 4) if peft_short_f1 else 0.0,
            },
        },
        "classification": {
            "count": len(base_cls_gold),
            "labels": sorted(cls_labels_union),
            "base": compute_classification_metrics(
                base_cls_gold, base_cls_pred, sorted(cls_labels_union)
            ) if base_cls_gold else {},
            "peft": compute_classification_metrics(
                peft_cls_gold, peft_cls_pred, sorted(cls_labels_union)
            ) if peft_cls_gold else {},
        },
        "details": details,
    }

    with open(OUT_REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report["short_qa"], ensure_ascii=False, indent=2))
    print(json.dumps(report["classification"], ensure_ascii=False, indent=2))
    print(f"Saved report to: {OUT_REPORT_JSON}")


if __name__ == "__main__":
    main()