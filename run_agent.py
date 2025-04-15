from agent.engine import execute_sandboxed_code
from agent.model import chat_with_model
from pydantic import BaseModel

class Story(BaseModel):
    title: str
    content: str

def run_agent():
    prompt = "What is the capital of France?"
    response, chat = chat_with_model(prompt)
    print(response)
    prompt_2 = "Create a story about the capital of France"
    response, chat = chat_with_model(prompt_2, chat, schema=Story)
    print(response)

def execute_code():
    # Example usage
    sample_code = """
def sum_of_numbers(list_of_numbers: list[int]) -> int:
    return sum(list_of_numbers)

a = sum_of_numbers([1, 2, 3, 4, 5])
b = a + 2
"""
    
    result, error = execute_sandboxed_code(sample_code)
    if error:
        print(f"Error: {error}")
    else:
        print(f"Result: {result}")

if __name__ == "__main__":
    #run_agent()
    execute_code()