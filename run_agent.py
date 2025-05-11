from agent.engine import execute_sandboxed_code
from agent.agent import Agent
from agent.settings import MEMORY_PATH
from agent.utils import create_memory_if_not_exists

def run_agent():
    agent = Agent()
    response = agent.chat("Hard to believe I'm already 28. I'm not sure how I feel about it. What feels good though, is I will have my favourite dessert today: chocolate cake.")
    print(response)

if __name__ == "__main__":
    run_agent()
    #execute_code()