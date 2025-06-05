from data.pipeline import generate_personas, generate_kb, generate_introduce_sft, generate_update_sft, generate_retrieve_sft
from data.utils import load_kb_from_json
import asyncio

async def main():
    """
    scenario = "Groningen, Netherlands in 2025"
    personas = generate_personas(8, scenario, save=True)
    kb = generate_kb(personas, save=True)
    await generate_introduce_sft(kb, num_turns=4)
    """
    kb = load_kb_from_json()
    await generate_retrieve_sft(kb, num_turns=4)
    #generate_introduce_sft(kb, num_turns=4)
    #generate_update_sft(kb, num_turns=4)

if __name__ == "__main__":
    asyncio.run(main())
