from data.pipeline import generate_personas, generate_kb, generate_introduce_sft, generate_update_sft, generate_retrieve_sft
from data.utils import load_kb_from_json

import asyncio
import time

async def main():
    """
    scenario = "Groningen, Netherlands in 2025"
    personas = generate_personas(4, scenario, save=True)
    kb = generate_kb(personas, save=True)
    """
    kb = load_kb_from_json()
    
    # Create tasks for all SFT generations to run concurrently
    tasks = [
        generate_retrieve_sft(kb, num_turns=4),
        generate_introduce_sft(kb, num_turns=4),
        generate_update_sft(kb, num_turns=4)
    ]
    
    # Start timer
    start_time = time.time()
    
    # Run all tasks concurrently and wait for them to complete
    await asyncio.gather(*tasks)
    
    # End timer and print elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = elapsed_time // 3600
    minutes = (elapsed_time % 3600) // 60
    seconds = elapsed_time % 60
    print(f"SFT generation completed in {hours:.0f}h {minutes:.0f}m {seconds:.2f}s")

if __name__ == "__main__":
    asyncio.run(main())
