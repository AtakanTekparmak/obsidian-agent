from data.pipeline import generate_personas, generate_momentary_stories, generate_kb, generate_multiturn_convos


def main():
    scenario = "Groningen, Netherlands in 2025"
    personas = generate_personas(4, scenario, save=True)
    stories = generate_momentary_stories(8, personas, save=True)
    kb = generate_kb(personas, stories, save=True)
    multiturn_convos = generate_multiturn_convos(kb, save=True) 

if __name__ == "__main__":
    main()