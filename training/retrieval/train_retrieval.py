import verifiers as vf
from training.retrieval import (
    create_kb_with_personas,
    build_verifiers_dataset,
    get_retrieval_rubric,
    RetrievalEnv,
)

"""
inference:
CUDA_VISIBLE_DEVICES=0,1,2,3 vf-vllm --model Qwen/Qwen1.5-7B-Chat --tensor-parallel-size 4 --max-batch-size 128

training:
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --config-file configs/zero3.yaml --num-processes 4 training/retrieval/train_retrieval.py
"""


def main():
    # Generate a KB and dataset for retrieval
    scenario = "friends planning a vacation"
    kb = create_kb_with_personas(num_personas=8, scenario=scenario)
    dataset = build_verifiers_dataset(kb)

    rubric = get_retrieval_rubric()
    env = RetrievalEnv(dataset=dataset, rubric=rubric)

    model_name = "Qwen/Qwen1.5-7B-Chat"
    model, tokenizer = vf.get_model_and_tokenizer(model_name)

    args = vf.grpo_defaults(run_name="retrieval_rl")
    args.num_iterations = 2
    args.per_device_train_batch_size = 8
    args.gradient_accumulation_steps = 4
    args.num_generations = 8

    trainer = vf.GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        env=env,
        args=args,
    )
    trainer.train()


if __name__ == "__main__":
    main()
