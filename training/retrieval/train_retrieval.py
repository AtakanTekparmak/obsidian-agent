import verifiers as vf
from training.retrieval import (
    build_verifiers_dataset,
    get_retrieval_rubric,
    RetrievalEnv,
)
from data.utils import load_kb_from_json

"""
inference:
CUDA_VISIBLE_DEVICES=0,1,2,3 vf-vllm --model Qwen/Qwen3-8B --tensor-parallel-size 4 --max-batch-size 128

training:
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --config-file verifiers/configs/zero3.yaml --num-processes 4 training/retrieval/train_retrieval.py
"""


def main():
    # Load KB from saved file and generate dataset for retrieval
    print("Building dataset for training with verifiers...")
    kb = load_kb_from_json()
    dataset = build_verifiers_dataset(kb)

    rubric = get_retrieval_rubric()
    env = RetrievalEnv(dataset=dataset, rubric=rubric)

    model_name = "Qwen/Qwen3-8B"
    model, tokenizer = vf.get_model_and_tokenizer(model_name)

    args = vf.grpo_defaults(run_name="retrieval_rl")
    args.num_iterations = 2
    args.per_device_train_batch_size = 8
    args.gradient_accumulation_steps = 4
    args.num_generations = 8

    print("Starting training with verifiers...")
    trainer = vf.GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        env=env,
        args=args,
    )
    trainer.train()


if __name__ == "__main__":
    main()
