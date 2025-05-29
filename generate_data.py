from data.pipeline import generate_personas, generate_kb, generate_introduce_sft, generate_update_sft
from data.utils import load_kb_from_json

from data.pipeline.generate_introduce_sft import generate_convo_for_persona_and_fact

def main():
    """
    scenario = "Groningen, Netherlands in 2025"
    personas = generate_personas(8, scenario, save=True)
    kb = generate_kb(personas, save=True)
    generate_introduce_sft(kb, num_turns=4)
    """
    kb = load_kb_from_json()
    generate_update_sft(kb, num_turns=4)

if __name__ == "__main__":
    main()