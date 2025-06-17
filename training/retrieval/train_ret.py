import verifiers as vf
from training.retrieval.memory_env import MemoryEnv
from training.retrieval.memory_rubric import MemoryRubric
from training.retrieval.dataset import load_verifiers_dataset
from data.utils import load_kb_from_json

"""
inference:
CUDA_VISIBLE_DEVICES=0,1,2,3 vf-vllm --model Qwen/Qwen3-8B --tensor-parallel-size 4 --max-batch-size 128

training:
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --config-file verifiers/configs/zero3.yaml --num-processes 4 training/retrieval/train_retrieval.py
"""


def main():
    # Load verifiers dataset
    dataset = load_verifiers_dataset()

    print("Instantiating rubric and environment...")
    env = MemoryEnv(dataset=dataset)

    print("Instantiating model...")
    model_name = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    model, tokenizer = vf.get_model_and_tokenizer(model_name, use_liger=False)

    args = vf.grpo_defaults(run_name="retrieval_rl")
    args.num_iterations = 2
    args.per_device_train_batch_size = 8
    args.gradient_accumulation_steps = 4
    args.num_generations = 8
    args.async_generation_timeout = 1200.0   # 20 minutes

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
