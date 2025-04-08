import os

from datagen.schemas.customer import CustomersList
from datagen.settings import OUTPUT_PATH, STORIES_PATH, GEMINI_MODEL
from datagen.schemas.stories import CustomerStories, CustomerStoriesList
from datagen.model import get_model_response
from datagen.utils import save_pydantic_to_json

def generate_user_stories(
        num_stories_per_customer: int,
        customers_list: CustomersList,
        save: bool = True
    ) -> CustomerStoriesList:
    """
    Generate user stories for each customer in the customers list.

    Args:
        customers_list: A list of customers
        save: Whether to save the user stories to a file

    Returns:
        A list of customer stories
    """
    customer_stories_list = []
    print("Generating user stories...")
    for customers in customers_list.customers:
        print(f"\t - Generating user stories for company {customers.company_name}...")
        prompt = f"Generate user stories for the following customers for the company {customers.company_name}:\n\n {customers.model_dump_json()}. Make sure to generate user stories that are relevant to the company's industry, products and services, and to the customer's bio and other details. Make sure to generate {num_stories_per_customer} user story(s) for each customer. The user stories should contain a customer experiencing an issue regarding a product/products of the company that they are using. The problem(s) the user is experiencing should be feasible given the context of the company, products and the customer data. The problem(s) should be of the kind that could be solved in one customer service conversation, and shouldn't be of the kind that would require a feature to be added to the product."
        response: CustomerStories = get_model_response(CustomerStories, prompt, GEMINI_MODEL)
        customer_stories_list.append(response)

    customer_stories_list = CustomerStoriesList(customer_stories=customer_stories_list)

    if save:
        output_path = os.path.join(OUTPUT_PATH, STORIES_PATH)
        os.makedirs(output_path, exist_ok=True)
        file_path = os.path.join(output_path, "user_stories.json")
        save_pydantic_to_json(customer_stories_list, file_path)

    return customer_stories_list
    
    
    