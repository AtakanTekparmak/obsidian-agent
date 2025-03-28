import asyncio

from data.pipeline.generate_kb import generate_personas

async def main():
    await generate_personas()

if __name__ == "__main__":
    asyncio.run(main())