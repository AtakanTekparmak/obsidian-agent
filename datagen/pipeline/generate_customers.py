import os 

from datagen.settings import OUTPUT_PATH, CUSTOMERS_PATH, GEMINI_MODEL
from datagen.schemas.customer import Customers, CustomersList
from datagen.schemas.company import Companies
from datagen.model import get_model_response
from datagen.utils import save_pydantic_to_json

def generate_customers(
        num_customers_per_company: int,
        companies: Companies,
        save: bool = True
    ) -> Customers:
    """
    Generate a list of customers for each company.

    Args:
        num_customers_per_company: The number of customers to generate for each company
        companies: A list of companies
        save: Whether to save the customers to a file

    Returns:
        A list of customers
    """
    print("Generating customers...")
    customers_list = []
    for company in companies.companies:
        prompt = f"Generate {num_customers_per_company} customers for the company:\n {company.model_dump_json()}. Make sure to generate customers that are relevant to the company's industry, products and services. Make sure to have a detailed user bio for each customer. Make sure that each customer uses at least one of the products offered by the company."
        response: Customers = get_model_response(Customers, prompt, GEMINI_MODEL)
        customers_list.append(response)

    customers_list = CustomersList(customers=customers_list)

    if save:
        output_path = os.path.join(OUTPUT_PATH, CUSTOMERS_PATH)
        os.makedirs(output_path, exist_ok=True)
        file_path = os.path.join(output_path, "customers.json")
        save_pydantic_to_json(customers_list, file_path)

    return customers_list