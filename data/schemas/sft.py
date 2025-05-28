import os

from data.schemas.base import BaseSchema

from agent.settings import MEMORY_PATH
from agent.utils import create_memory_if_not_exists

class StaticMemory(BaseSchema):
    guideline: str
    user_file_path: str
    user_file_content: str

    def instantiate(self):
        """
        Instantiate the static memory inside the memory path.
        """
        create_memory_if_not_exists()
        try:
            guideline_path = os.path.join(MEMORY_PATH, "guideline.md")
            with open(guideline_path, "w") as f:
                f.write(self.guideline)
            user_file_path = os.path.join(MEMORY_PATH, self.user_file_path)
            user_dir = os.path.dirname(user_file_path)
            os.makedirs(user_dir, exist_ok=True)
            with open(user_file_path, "w") as f:
                f.write(self.user_file_content)
        except Exception as e:
            print(f"Error instantiating static memory: {e}")
            raise e
            