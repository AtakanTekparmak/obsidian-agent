from agent.engine import execute_sandboxed_code
from agent.model import get_model_response
from agent.settings import MEMORY_PATH
from agent.utils import create_memory_if_not_exists

from pydantic import BaseModel

class Story(BaseModel):
    title: str
    content: str

def run_agent():
    prompt = "What is the capital of France?"
    response = get_model_response(message=prompt)
    print(response)
    prompt_2 = "Create a story about the capital of France"
    response = get_model_response(message=prompt_2, schema=Story)
    print(response)

def execute_code():
    # Example usage
    sample_code = """
def sum_of_numbers(list_of_numbers: list[int]) -> int:
    return sum(list_of_numbers)

def something_else(a: int, b: int) -> int:
    c = a + b
    return c * 2

a = sum_of_numbers([1, 2, 3, 4, 5])
b = a + 2
c = something_else(a, b)
d = create_file("test.txt", "Hello, world!")
e = create_dir("test_dir")
f = write_to_file("test.txt", "Hello, everyone!")
g = read_file("test.txt")
j = create_file("test_dir/a.txt", "Hello, world!")
k = list_files()
"""
    create_memory_if_not_exists()
    result, error = execute_sandboxed_code(
        sample_code,
        import_module="agent.tools",
        allowed_path=MEMORY_PATH
    )
    if error:
        print(f"Error: {error}")
    else:
        print(f"Result: {result}")

if __name__ == "__main__":
    #run_agent()
    execute_code()