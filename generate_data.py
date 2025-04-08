#from data.pipeline import generate_personas, generate_momentary_stories, generate_kb, generate_qa
from datagen.pipeline.generate_companies import generate_companies
from datagen.pipeline.generate_customers import generate_customers

"""
def main():
    scenario = "Amsterdam, Netherlands in 2024"
    personas = generate_personas(8, scenario, save=True)
    stories = generate_momentary_stories(16, personas, save=True)
    kb = generate_kb(personas, stories, save=True)
    qa = generate_qa(10, kb, save=True)
"""
def main():
    companies = generate_companies(2, save=True)
    customers = generate_customers(2, companies, save=True)
if __name__ == "__main__":
    main()