from __future__ import annotations
import json, re, argparse
from pathlib import Path
from datasets import Dataset, concatenate_datasets, load_dataset
from huggingface_hub import HfApi

ID_RE   = re.compile(r"convo_(.+)\.json")     # captures the unique id
COLUMNS = ["conversations", "task_type", "id"]
DATASET_NAME = "AtakanTekparmak/obsidian-agent-sft-xml"

def scan_data(root: Path) -> Dataset:
    """Walk sub-folders and return a Dataset with the mandatory columns."""
    rows = []
    for sub in ["introduce", "update", "retrieve"]:
        for fp in (root / sub).glob("convo_*.json"):
            uid_match = ID_RE.fullmatch(fp.name)
            if not uid_match:
                continue
            with fp.open(encoding="utf-8") as f:
                payload = json.load(f)                 # list[dict[…]]
            rows.append(
                {
                    "conversations": json.dumps(payload, ensure_ascii=False),
                    "task_type": sub,
                    "id": uid_match.group(1),
                }
            )
    return Dataset.from_list(rows)                  

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, type=Path)
    parser.add_argument("--repo_id", default=DATASET_NAME)
    args = parser.parse_args()

    local_ds = scan_data(args.data_dir)

    api = HfApi()
    repo_exists = api.repo_exists(args.repo_id, repo_type="dataset")      # ≈ HEAD request

    if repo_exists:
        # Load the current remote split and merge while de-duplicating on `id`
        remote_ds = load_dataset(args.repo_id, split="train", trust_remote_code=True)
        concatenated = concatenate_datasets([remote_ds, local_ds])
        merged = concatenated.to_pandas().drop_duplicates(subset=['id']).reset_index(drop=True)
        merged = Dataset.from_pandas(merged)
    else:
        merged = local_ds

    merged.push_to_hub(
        args.repo_id,
        commit_message="Add new conversations",
    )

if __name__ == "__main__":
    main()