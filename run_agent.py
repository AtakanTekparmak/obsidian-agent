from xml_agent.agent import Agent as XmlAgent

def run_xml_agent():
    agent = XmlAgent()
    response = agent.chat("Hard to believe I'm already 28. I'm not sure how I feel about it. What feels good though, is I will have my favourite dessert today: chocolate cake.")
    response_2 = agent.chat("You know what is great about growing older though? There's less expectations on you.")
    agent.save_conversation(save_folder="xml")
    
if __name__ == "__main__":
    run_xml_agent()