from data.pipeline import generate_personas, generate_momentary_stories

def main():
    scenario = "Amsterdam, Netherlands in 2024"
    personas = generate_personas(8, scenario, save=True)
    stories = generate_momentary_stories(8, personas, save=True)
    print(stories)

if __name__ == "__main__":
    main()