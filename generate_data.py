import asyncio

from data.pipeline import generate_personas, generate_relationships
from data.settings import SCENARIO

async def main():
    backstories = await generate_personas(scenario=SCENARIO)
    relationships = await generate_relationships(backstories=backstories)
if __name__ == "__main__":
    asyncio.run(main())