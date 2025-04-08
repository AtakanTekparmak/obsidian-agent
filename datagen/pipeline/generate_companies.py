import os

from datagen.settings import OUTPUT_PATH, COMPANIES_PATH, GEMINI_MODEL
from datagen.schemas.company import Companies
from datagen.model import get_model_response
from datagen.utils import save_pydantic_to_json

def generate_companies(
        num_companies: int,
        save: bool = True
    ) -> Companies:
    """
    Create a list of companies based on a scenario.

    Args:
        num_companies: The number of companies to generate
        save: Whether to save the companies to a file

    Returns:
        A list of companies
    """
    print("Generating companies...")
    prompt = f"Create {num_companies} companies based on the provided schema. Make sure to generate data for Business-to-Customer (B2C) companies that offer customer support services on their products and customer issues. The business should be able to offer digital customer support and solve problems in that medium. Make sure to have 1-3 products offered by each company."
    response: Companies = get_model_response(Companies, prompt, GEMINI_MODEL)

    if save:
        output_path = os.path.join(OUTPUT_PATH, COMPANIES_PATH)
        os.makedirs(output_path, exist_ok=True)
        file_path = os.path.join(output_path, "companies.json")
        save_pydantic_to_json(response, file_path)

    return response