import pandas as pd
from pathlib import Path
import json
#处理数据集
DATA_DIR = Path(".")  # 当前目录
QUESTION_FILE ="cMedQA/questions.csv/questions.csv"
ANSWER_FILE = "cMedQA/answers.csv/answers.csv"
TRAIN_FILE = "cMedQA/train_candidates.txt/train_candidates.txt"
DEV_FILE = "cMedQA/dev_candidates.txt/dev_candidates.txt"
TEST_FILE ="cMedQA/test_candidates.txt/test_candidates.txt"


def clean_columns(df):
    df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=False)
    return df


def load_base_data():
    questions = pd.read_csv(QUESTION_FILE, encoding="utf-8")
    answers = pd.read_csv(ANSWER_FILE, encoding="utf-8")

    questions = clean_columns(questions)
    answers = clean_columns(answers)

    questions = questions.rename(columns={
        "que_id": "question_id",
        "content": "question"
    })

    answers = answers.rename(columns={
        "ans_id": "answer_id",
        "content": "answer"
    })

    questions["question_id"] = questions["question_id"].astype(str)
    answers["answer_id"] = answers["answer_id"].astype(str)

    return questions, answers


def build_sft(split_file, output_name):
    questions, answers = load_base_data()

    candidates = pd.read_csv(split_file, encoding="utf-8")
    candidates = clean_columns(candidates)

    print(f"\n处理文件: {split_file}")
    print("原始列名:", candidates.columns.tolist())

    # ===== 统一 question_id =====
    if "question_id" not in candidates.columns:
        if "que_id" in candidates.columns:
            candidates = candidates.rename(columns={"que_id": "question_id"})
        else:
            raise ValueError(f"找不到 question_id 列: {candidates.columns.tolist()}")

    # ===== 统一 answer_id =====
    if "pos_ans_id" in candidates.columns:
        # train 格式
        candidates = candidates.rename(columns={"pos_ans_id": "answer_id"})

    elif "ans_id" in candidates.columns:
        # dev / test 格式
        candidates = candidates.rename(columns={"ans_id": "answer_id"})

    elif "answer_id" in candidates.columns:
        pass

    else:
        raise ValueError(f"找不到 answer_id 列: {candidates.columns.tolist()}")

    # ===== 只保留正样本（dev/test）=====
    if "label" in candidates.columns:
        candidates = candidates[candidates["label"] == 1]

    # ===== 只保留必要列 =====
    candidates = candidates[["question_id", "answer_id"]].drop_duplicates()

    # 类型统一
    candidates["question_id"] = candidates["question_id"].astype(str)
    candidates["answer_id"] = candidates["answer_id"].astype(str)

    # ===== 合并问题 =====
    df = candidates.merge(
        questions[["question_id", "question"]],
        on="question_id",
        how="inner"
    )

    # ===== 合并答案 =====
    df = df.merge(
        answers[["answer_id", "answer"]],
        on="answer_id",
        how="inner"
    )

    # ===== 清洗 =====
    df = df.dropna(subset=["question", "answer"])
    df["question"] = df["question"].astype(str).str.strip()
    df["answer"] = df["answer"].astype(str).str.strip()

    df = df[df["question"].str.len() >= 5]
    df = df[df["answer"].str.len() >= 20]
    df = df.drop_duplicates(subset=["question", "answer"])

    # ===== 转 SFT =====
    sft = pd.DataFrame({
        "instruction": "请根据患者描述进行医学分析，并给出专业、简洁、准确的回答。",
        "input": df["question"],
        "output": df["answer"]
    })

    # ===== 保存 =====
    json_path = DATA_DIR / f"{output_name}.json"
    jsonl_path = DATA_DIR / f"{output_name}.jsonl"

    sft.to_json(json_path, orient="records", force_ascii=False, indent=2)
    sft.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)

    print(f"{output_name} 生成完成")
    print("数据量:", len(sft))


if __name__ == "__main__":
    build_sft(TRAIN_FILE, "cmedqa_train_sft")
    build_sft(DEV_FILE, "cmedqa_dev_sft")
    build_sft(TEST_FILE, "cmedqa_test_sft")