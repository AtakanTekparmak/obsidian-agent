from data.pipeline import generate_personas, generate_momentary_stories, generate_kb, generate_multiturn_convos
from datagen.pipeline.generate_companies import generate_companies
from datagen.pipeline.generate_customers import generate_customers
from datagen.pipeline.generate_user_stories import generate_user_stories


def main():
    scenario = "Amsterdam, Netherlands in 2024"
    personas = generate_personas(4, scenario, save=True)
    stories = generate_momentary_stories(8, personas, save=True)
    kb = generate_kb(personas, stories, save=True)
    multiturn_convos = generate_multiturn_convos(kb, save=True) 
"""
def main():
    companies = generate_companies(2, save=True)
    customers = generate_customers(2, companies, save=True)
    user_stories = generate_user_stories(1, customers, save=True)
"""

if __name__ == "__main__":
    main()