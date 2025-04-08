from typing import List

from datagen.schemas.base import BaseSchema

class Product(BaseSchema):
    name: str
    description: str
    scope: str
    features: List[str]

class Company(BaseSchema):
    name: str
    industry_general: str
    industry_specific: str
    customer_profile: str   
    products_offered: List[Product]

class Companies(BaseSchema):
    companies: List[Company]