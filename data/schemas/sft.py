import os
import shutil

from data.schemas.base import BaseSchema

from agent.utils import create_memory_if_not_exists

class EntityFile(BaseSchema):
    entity_name: str
    entity_file_path: str
    entity_file_content: str

class StaticMemory(BaseSchema):
    user_md: str
    entities: list[EntityFile]

    def instantiate(self, path: str):
        """
        Instantiate the static memory inside the memory path.
        """
        try:
            # Create the base memory directory
            create_memory_if_not_exists(path)
            
            # Write user.md file
            user_md_path = os.path.join(path, "user.md")
            with open(user_md_path, "w") as f:
                f.write(self.user_md)
            
            # Write entity files
            for entity in self.entities:
                entity_file_path = os.path.join(path, entity.entity_file_path)
                
                # Ensure parent directory exists
                entity_dir = os.path.dirname(entity_file_path)
                if entity_dir and not os.path.exists(entity_dir):
                    os.makedirs(entity_dir, exist_ok=True)
                
                # Write the entity file
                with open(entity_file_path, "w") as f:
                    f.write(entity.entity_file_content)
                    
        except Exception as e:
            print(f"Error instantiating static memory at {path}: {e}")
            raise

    def reset(self, path: str):
        """
        Reset the static memory inside the memory path.
        """
        try:
            # Delete the memory directory
            if os.path.exists(path):
                shutil.rmtree(path)

            # Call the instantiate method
            self.instantiate(path)
        except Exception as e:
            print(f"Error resetting static memory at {path}: {e}")
            raise

class FactUpdate(BaseSchema):
    initial_fact: str
    updated_fact: str
    fact_update_possible: bool