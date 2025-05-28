from data.pipeline import generate_personas, generate_kb, generate_sft, generate_static_memory
from data.utils import load_kb_from_json

def main():
    """
    scenario = "Groningen, Netherlands in 2025"
    personas = generate_personas(8, scenario, save=True)
    kb = generate_kb(personas, save=True)
    generate_sft(kb, num_turns=4)
    static_memory = generate_static_memory(
        persona=kb.items[0].persona, 
        fact=kb.items[0].facts[0].fact_description
    )
    """
    kb = load_kb_from_json()
    generate_sft(kb, num_turns=4)

if __name__ == "__main__":
    main()