from data.schemas.base import BaseSchema

class StaticMemory(BaseSchema):
    guideline: str
    user_file_path: str
    user_file_content: str