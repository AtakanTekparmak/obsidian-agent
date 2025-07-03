import asyncio

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


def capture_xml_tag(text, tag):
    pattern = rf"<{tag}([^>]*)>(.*?)</{tag}>"
    try:
        return re.findall(pattern, text, re.DOTALL)[0][1].strip()
    except IndexError:
        print(f"Tag <{tag}> not found in the text.")
        return None


def list_folders(path):
    return [
        name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))
    ]


# Pydantic model for JSONL entries
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


def read_jsonl(
    path, category: Literal["retrieval", "clarification", "update"] = "retrieval"
):
    if category != "update":
        with open(path, "r", encoding="utf-8") as f:
            return [QAEntry(**json.loads(line)) for line in f]
    else:
        with open(path, "r", encoding="utf-8") as f:
            return [UpdateQAEntry(**json.loads(line)) for line in f]


MODEL_NAME = "google/gemini-2.5-pro"  # anthropic/claude-sonnet-4 #qwen/qwen3-14b #google/gemma-3-27b-it #google/gemma-3-12b-it #"google/gemini-2.5-pro"
JUDGE_NAME = "google/gemini-2.5-pro"
TMP_DIR = "tmp_work"


async def evaluate_agents():
    """
    Run agents on the retrieval evaluation dataset.
    """
    categories = ["retrieval", "clarification", "update"]
    evaluation_table = {"agent": MODEL_NAME, "avg": 0.0, "scores": []}
    os.makedirs(TMP_DIR, exist_ok=True)
    for category in categories:
        base_path = f"data/eval/{category}"
        # list the folder names
        if not os.path.exists(base_path):
            print(f"Base path does not exist: {base_path}")
            raise FileNotFoundError(f"Base path does not exist: {base_path}")

        folders = list_folders(base_path)
        print(f"Found folders: {folders}")

        for folder in tqdm(folders, desc="Folders"):
            src_folder_path = os.path.join(base_path, folder)
            tmp_folder_path = os.path.join(TMP_DIR, folder)
            if os.path.exists(tmp_folder_path):
                print(f"Removing existing temporary folder: {tmp_folder_path}")
                shutil.rmtree(tmp_folder_path)
            src_qa_path = os.path.join(base_path, f"{folder}_qa.jsonl")
            tmp_qa_path = os.path.join(TMP_DIR, f"{folder}_qa.jsonl")
            if os.path.exists(src_folder_path):
                shutil.copytree(src_folder_path, tmp_folder_path, dirs_exist_ok=True)
                shutil.copy2(src_qa_path, tmp_qa_path)
                agent = Agent(model=MODEL_NAME, memory_path=tmp_folder_path)
                qa = read_jsonl(tmp_qa_path, category)

                for entry in tqdm(qa, desc=f"Entries in {folder}", leave=False):
                    if category == "update":
                        agent.chat(entry.update)
                        retrieval_agent = Agent(
                            model=MODEL_NAME, memory_path=tmp_folder_path
                        )
                        retrieval_agent.chat(entry.question)
                        model_answer = retrieval_agent.messages[-1]
                        answer = extract_reply(model_answer.content)
                        """
                        print(
                            json.dumps(
                                [a.content for a in agent.messages[1:]],
                                indent=4,
                                ensure_ascii=False,
                            )
                        )
                        print("**Model Anwswer**")
                        print(
                            json.dumps(
                                [a.content for a in retrieval_agent.messages[1:]],
                                indent=4,
                                ensure_ascii=False,
                            )
                        )
                        print("**" * 10)
                        """
                        # reset memory, keep the first message
                        agent.messages = [agent.messages[0]]
                        shutil.rmtree(tmp_folder_path)
                        shutil.copytree(
                            src_folder_path, tmp_folder_path, dirs_exist_ok=True
                        )
                        shutil.copy2(src_qa_path, tmp_qa_path)
                    else:
                        agent.messages = [agent.messages[0]]
                        agent.chat(entry.question)
                        print(
                            json.dumps(
                                [a.content for a in agent.messages[1:]],
                                indent=4,
                                ensure_ascii=False,
                            )
                        )
                        model_answer = agent.messages[-1]
                        answer = extract_reply(model_answer.content)

                    jp = JUDGE_PROMPT.render(
                        question=entry.question,
                        correct_answer=entry.answer,
                        answer=answer,
                        judge=entry.judge,
                    )
                    response = await get_model_response(
                        message=jp, use_vllm=False, model=JUDGE_NAME
                    )
                    is_correct = capture_xml_tag(response, tag="judgment")
                    evaluation_table["scores"].append(
                        {
                            "question": entry.question,
                            "answer": model_answer.content,
                            "correct_answer": entry.answer,
                            "judge": entry.judge,
                            "is_correct": is_correct,
                            "category": category,
                        }
                    )
                agent.save_conversation(save_folder=tmp_folder_path)
            else:
                print(f"Folder not found: {src_folder_path}")

    for score in evaluation_table["scores"]:
        if score["is_correct"] is None:
            print(f"Warning: No judgment found for question: {score['question']}")
            score["is_correct"] = "JUDGE_FAILED"

        if score["is_correct"].lower() == "correct":
            evaluation_table["avg"] += 1.0
        else:
            evaluation_table["avg"] += 0.0
    evaluation_table["avg"] /= len(evaluation_table["scores"])

    with open("evaluation_results.json", "w") as f:
        json.dump(evaluation_table, f, indent=4)


if __name__ == "__main__":
    asyncio.run(evaluate_agents())
