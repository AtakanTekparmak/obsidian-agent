import json

from training.vf.env import MemoryEnv
from data.schemas.sft import StaticMemory

import verifiers as vf

# Constants
STATIC_MEMORY_PATH = "memory/base_agent_memory/"
BASE_MEMORY_JSON_PATH = "output/static_mem/base_memory.json"

MODEL_NAME = "Qwen/Qwen3-14B"

def create_static_memory():
    """
    Create the static memory.
    """
    try:
        # Load the static memory from the JSON file
        with open(BASE_MEMORY_JSON_PATH, "r") as f:
            static_memory = StaticMemory.model_validate_json(f.read())

        # Instantiate the static memory
        static_memory.instantiate(STATIC_MEMORY_PATH)
    except Exception as e:
        print(f"Error creating static memory: {e}")
        raise
    finally:
        # Reset the static memory
        static_memory.reset(STATIC_MEMORY_PATH)
    
def main():
    # Create the static memory
    create_static_memory()

    # Create the environment
    vf_env = MemoryEnv(memory_path=STATIC_MEMORY_PATH)

    # Get the model and tokenizer
    model, tokenizer = vf.get_model_and_tokenizer(MODEL_NAME)

    # Construct the run name
    run_name = f"vf-ret-{MODEL_NAME}-1"

    # Construct the args
    args = vf.grpo_defaults(run_name=run_name)
    args.num_train_epochs = 200

    # Construct the trainer
    trainer = vf.GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        env=vf_env,
        args=args,
    )
    trainer.train()

if __name__ == "__main__":
    main()
