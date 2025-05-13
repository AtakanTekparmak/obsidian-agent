from typing import List, Dict
import json
import re
import os
import tempfile
import subprocess

from verifiers.parsers import XMLParser
from verifiers.rubrics import Rubric
from verifiers.envs.memory_agent.model import get_model_response

class MemoryRubric(Rubric):
    def __init__(
            self,
            parser: XMLParser = XMLParser(fields=["thoughts", ("python", "answer")]),
            env_parser: XMLParser = XMLParser(fields=["result"]),
        ):
        self.parser = parser
        self.env_parser = env_parser
        self.reward_funcs = [
            self.parser.get_xml_reward_func(),
            self.check_facts_reward_func
        ]
        self.reward_weights = [0.3, 0.7]  # Emphasize fact checking over XML formatting
        
        # Load judge prompt
        current_dir = os.path.dirname(os.path.abspath(__file__))
        judge_prompt_path = os.path.join(current_dir, "..", "envs", "memory_agent", "judge_prompt.txt")
        with open(judge_prompt_path, "r") as f:
            self.judge_prompt = f.read()
    
    def get_reward_funcs(self) -> List:
        return self.reward_funcs
    
    def get_reward_weights(self) -> List[float]:
        return self.reward_weights
    
    def extract_memory_dump(self, messages: List[Dict[str, str]]) -> str:
        """
        Extracts the memory dump from the messages.
        
        Args:
            messages: The list of messages.
            
        Returns:
            The memory dump as a string.
        """
        dump = ""
        for msg in messages:
            if msg["role"] == "assistant":
                parsed = self.parser.parse(msg["content"])
                if hasattr(parsed, "python") and parsed.python is not None:
                    # Extract the memory dump by running folder_dump.py on the memory path
                    dump_code = parsed.python
                    if "list_files" in dump_code or "read_file" in dump_code:
                        # This code might read files - use it to collect memory state
                        dump += f"\n\nCode execution: {dump_code}\n"
                        
                        # Check if there's a corresponding user message with the result
                        msg_idx = messages.index(msg)
                        if msg_idx + 1 < len(messages) and messages[msg_idx + 1]["role"] == "user":
                            result_msg = messages[msg_idx + 1]
                            parsed_result = self.env_parser.parse(result_msg["content"])
                            if hasattr(parsed_result, "result") and parsed_result.result:
                                dump += f"\nResult: {parsed_result.result}\n"
        
        return dump
        
    def check_facts_reward_func(
            self,
            completions: List[List[Dict[str, str]]],
            facts_to_check_so_far: List[Dict[str, str]] = None,
            **kwargs
    ) -> List[float]:
        """
        Reward function that checks if the facts_to_check_so_far are present in the memory.
        
        Args:
            completions: The list of completions.
            facts_to_check_so_far: The list of facts to check.
            
        Returns:
            A list of rewards (0.0 to 1.0) based on how many facts are found.
        """
        rewards = []
        
        for completion in completions:
            if not facts_to_check_so_far:
                rewards.append(0.0)
                continue
                
            # Extract memory dump from the completion
            memory_dump = self.extract_memory_dump(completion)
            
            if not memory_dump:
                rewards.append(0.0)
                continue
            
            # Create a prompt for the judge to check facts
            judge_input = self.judge_prompt + f"\n<dump>\n{memory_dump}\n</dump>\n"
            
            # Add facts to check
            judge_input += f"\n<facts>\n\"facts_to_check_so_far\": {json.dumps(facts_to_check_so_far, indent=4)}\n</facts>"
            
            # Get model response using the judge
            try:
                response = get_model_response(message=judge_input)
                json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
                
                if json_match:
                    result = json.loads(json_match.group(1))
                    total_facts = result.get("total_facts_checked", 0)
                    facts_present = result.get("num_facts_present", 0)
                    
                    if total_facts > 0:
                        reward = facts_present / total_facts
                    else:
                        reward = 0.0
                else:
                    reward = 0.0
            except Exception as e:
                print(f"Error in check_facts_reward_func: {e}")
                reward = 0.0
                
            rewards.append(reward)
            
        return rewards
                    