from agent.agent import Agent
import uuid

MODEL_NAME = "google/gemini-2.5-flash"

def run_agent():
    agent = Agent(
        model=MODEL_NAME,
        memory_path = "memory/test_diff"
    )

    response = agent.chat("I'm 23 years old.")
    response_2 = agent.chat("I just turned 24, life's moving fast.")
    agent.save_conversation(save_folder="test_diff")
    
if __name__ == "__main__":
    run_agent()