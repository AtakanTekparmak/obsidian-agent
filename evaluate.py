import asyncio
import argparse

from pydantic import BaseModel
import os
import shutil
import json
from typing import Literal

from tqdm import tqdm
from agent.agent import Agent
from agent.async_agent.async_model import get_model_response
from judge import JUDGE_PROMPT
import re
from agent.utils import extract_reply

# Pydantic models for JSONL entries
class QAEntry(BaseModel):
    question: str
    answer: str
    judge: str = ""

    def __str__(self):
        return f"Question: {self.question}\nAnswer: {self.answer}\nJudge: {self.judge}"

class UpdateQAEntry(BaseModel):
    question: str
    answer: str
    judge: str = ""
    original: str
    diff: str
    update: str

    def __str__(self):
        return f"Question: {self.question}\nAnswer: {self.answer}\nJudge: {self.judge}\nOriginal: {self.original}\nDiff: {self.diff}\nUpdate: {self.update}"

# Helper functions

def capture_xml_tag(text, tag):
    pattern = rf"<{tag}([^>]*)>(.*?)</{tag}>"
    try:
        return re.findall(pattern, text, re.DOTALL)[0][1].strip()
    except IndexError:
        print(f"Tag <{tag}> not found in the text.")
        return None


def list_folders(path):
    return [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]


def read_jsonl(path, category: Literal["retrieval", "clarification", "update"] = "retrieval"):
    with open(path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]
    model = UpdateQAEntry if category == "update" else QAEntry
    return [model(**entry) for entry in lines]

async def evaluate_agents(model_name, judge_name, use_vllm, tmp_dir, data_dir):
    categories = ["retrieval", "clarification", "update"]
    evaluation = {"agent": model_name, "avg": 0.0, "scores": []}
    os.makedirs(tmp_dir, exist_ok=True)

    for category in categories:
        base_path = os.path.join(data_dir, category)
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"Base path not found: {base_path}")

        for folder in list_folders(base_path):
            src_folder = os.path.join(base_path, folder)
            tmp_folder = os.path.join(tmp_dir, folder)
            shutil.rmtree(tmp_folder, ignore_errors=True)
            shutil.copytree(src_folder, tmp_folder)

            src_qa = os.path.join(base_path, f"{folder}_qa.jsonl")
            tmp_qa = os.path.join(tmp_dir, f"{folder}_qa.jsonl")
            shutil.copy2(src_qa, tmp_qa)

            agent = Agent(model=model_name, memory_path=tmp_folder, use_vllm=use_vllm)
            qa_list = read_jsonl(tmp_qa, category)

            for entry in tqdm(qa_list, desc=f"{category}/{folder}"):
                agent.messages = agent.messages[:1]
                if category == "update":
                    agent.chat(entry.update)
                    retrieval_agent = Agent(model=model_name, memory_path=tmp_folder, use_vllm=use_vllm)
                    retrieval_agent.chat(entry.question)
                    model_answer = retrieval_agent.messages[-1]
                    answer = extract_reply(model_answer.content)
                else:
                    agent.chat(entry.question)
                    model_answer = agent.messages[-1]
                    answer = extract_reply(model_answer.content)

                jp = JUDGE_PROMPT.render(
                    question=entry.question,
                    correct_answer=entry.answer,
                    answer=answer,
                    judge=entry.judge,
                )
                response = await get_model_response(message=jp, use_vllm=False, model=judge_name)
                is_correct = capture_xml_tag(response, "judgment")

                evaluation["scores"].append({
                    "question": entry.question,
                    "answer": model_answer.content,
                    "correct": entry.answer,
                    "judge": entry.judge,
                    "is_correct": is_correct,
                    "category": category,
                })

            # cleanup for next folder
            shutil.rmtree(tmp_folder)

    # compute average
    total = len(evaluation["scores"])
    correct = sum(1 for s in evaluation["scores"] if str(s.get("is_correct")).lower() == "correct")
    evaluation["avg"] = correct / total if total else 0

    with open("evaluation_results.json", "w") as f:
        json.dump(evaluation, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate agents on QA datasets.")
    parser.add_argument("--model", default="qwen/qwen3-8b", help="Model name for agent")
    parser.add_argument("--judge", default="google/gemini-2.5-pro", help="Judge model name")
    parser.add_argument(
        "--use-vllm", action="store_true", help="Use vLLM for agent inference"
    )
    parser.add_argument(
        "--tmp-dir", default="tmp_work", help="Temporary working directory"
    )
    parser.add_argument(
        "--data-dir", default="data/eval", help="Base data directory containing categories"
    )
    args = parser.parse_args()

    asyncio.run(
        evaluate_agents(
            model_name=args.model,
            judge_name=args.judge,
            use_vllm=args.use_vllm,
            tmp_dir=args.tmp_dir,
            data_dir=args.data_dir,
        )
    )