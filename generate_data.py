from data.pipeline import generate_personas, generate_momentary_stories, generate_kb, generate_multiturn_convos, generate_sft

from data.utils import load_kb_from_json, load_convos_from_json


def main():
    """
    scenario = "Groningen, Netherlands in 2025"
    personas = generate_personas(6, scenario, save=True)
    stories = generate_momentary_stories(12, personas, save=True)
    kb = generate_kb(personas, stories, save=True)
    multiturn_convos = generate_multiturn_convos(kb, save=True) 
    """
    multiturn_convos = load_convos_from_json()
    generate_sft(multiturn_convos)

if __name__ == "__main__":
    main()