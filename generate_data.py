from data.pipeline import generate_personas

def main():
    scenario = "Amsterdam, Netherlands in 2024"
    personas = generate_personas(8, scenario, save=True)
    print(personas)

if __name__ == "__main__":
    main()