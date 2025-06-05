import os

from data.schemas.base import BaseSchema

from agent.settings import MEMORY_PATH
from agent.utils import create_memory_if_not_exists

class StaticMemory(BaseSchema):
    guideline: str
    user_file_path: str
    user_file_content: str

    def instantiate(self, path: str = MEMORY_PATH):
        """
        Instantiate the static memory inside the memory path.
        """
        create_memory_if_not_exists(path)
        try:
            guideline_path = os.path.join(path, "guideline.md")
            with open(guideline_path, "w") as f:
                f.write(self.guideline)
            user_file_path = os.path.join(path, self.user_file_path)
            user_dir = os.path.dirname(user_file_path)
            os.makedirs(user_dir, exist_ok=True)
            with open(user_file_path, "w") as f:
                f.write(self.user_file_content)
        except Exception as e:
            print(f"Error instantiating static memory: {e}")
            raise e
            

class FactUpdate(BaseSchema):
    initial_fact: str
    updated_fact: str
    fact_update_possible: bool