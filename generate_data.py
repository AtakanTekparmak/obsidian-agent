import asyncio

from data.pipeline.generate_kb import generate_personas
from data.settings import SCENARIO

async def main():
    await generate_personas(scenario=SCENARIO)

if __name__ == "__main__":
    asyncio.run(main())