from agent.agent import Agent
import uuid

MODEL_NAME = "mistralai/mistral-medium-3"

def run_agent():
    agent = Agent(
        model=MODEL_NAME,
        memory_path = f"memory_{uuid.uuid4()}",
        predetermined_memory_path=True
    )

    response = agent.chat("Hard to believe I'm already 28. I'm not sure how I feel about it. What feels good though, is I will have my favourite dessert today: chocolate cake.")
    response_2 = agent.chat("You know what is great about growing older though? There's less expectations on you.")
    agent.save_conversation(save_folder="new_prompt_test")
    
if __name__ == "__main__":
    run_agent()