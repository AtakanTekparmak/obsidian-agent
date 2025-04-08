from typing import List

from datagen.schemas.base import BaseSchema

class CustomerStory(BaseSchema):
    customer_id: str
    customer_name_surname: str
    user_story: str

class CustomerStories(BaseSchema):
    company_name: str
    customer_stories: List[CustomerStory]

class CustomerStoriesList(BaseSchema):
    customer_stories: List[CustomerStories]