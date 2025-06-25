from agent.agent import Agent
import uuid

MODEL_NAME = "google/gemini-2.5-flash"

def run_agent():
    agent = Agent(
        model=MODEL_NAME,
        memory_path = "memory/test_ret"
    )

    response = agent.chat("What's the age of my wife?")
    agent.save_conversation(save_folder="new_prompt_test")
    
if __name__ == "__main__":
    run_agent()