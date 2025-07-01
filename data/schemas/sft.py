import os

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
        create_memory_if_not_exists(path)
        user_md_path = os.path.join(path, "user.md")
        with open(user_md_path, "w") as f:
            f.write(self.user_md)
        for entity in self.entities:
            entity_file_path = os.path.join(path, entity.entity_file_path)
            with open(entity_file_path, "w") as f:
                f.write(entity.entity_file_content)

    def to_json(self):
        """
        Return a dictionary representation for JSON serialization.
        """
        return {
            "guideline": self.guideline,
            "user_file_path": self.user_file_path,
            "user_file_content": self.user_file_content,
        }


class FactUpdate(BaseSchema):
    initial_fact: str
    updated_fact: str
    fact_update_possible: bool
