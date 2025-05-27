from data.pipeline import generate_personas, generate_kb, generate_sft

def main():
    scenario = "Groningen, Netherlands in 2025"
    personas = generate_personas(6, scenario, save=True)
    kb = generate_kb(personas, save=True)
    generate_sft(kb, num_turns=4)
    """
    kb = load_kb_from_json()
    generate_sft(kb, num_turns=4)
    """

if __name__ == "__main__":
    main()