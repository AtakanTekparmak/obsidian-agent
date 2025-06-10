# Obsidian Agent Retrieval RL

## Overall Idea

1. Generate a kb with personas and facts for each persona
2. Generate a static memory using the `generate_static_memory` function in `data/pipeline/sft/generate_update_sft.py` for each fact in a persona
3. Generate a direct question for the agent to elicit the fact, for each fact in a persona, using a prompt similar to the `UPDATE_GEN_PROMPT` in `data/pipeline/sft/generate_update_sft.py` (but for generation a question instead of a fact. The function should be called `generate_question_prompt`).
4. Construct a verifiers dataset like so:
```python
dataset = [
    [
        {
            "prompt": generate_question_prompt(fact),
            "answer": fact,
            "task": "retrieval",
            "static_memory": generate_static_memory(persona, fact),
            "persona": persona.name_surname,
            "fact": fact.fact_description
        } for fact in persona.facts
    ] for persona in kb.personas
].flatten()
```
5. Generate a verifiers rubric using the `get_reward` function in `training/reward/reward.py` to assess if the fact is present in the agent's reply.
6. Generate a verifiers environment using the rubric, the dataset and the agent located in `agent/` (the full logic of the agent should be present in the environment) so we can then train an agent on the retrieval task.