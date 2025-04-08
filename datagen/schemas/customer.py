import enum
from typing import List

from data.schemas.base import BaseSchema

class Gender(str, enum.Enum):
    MALE = "male"
    FEMALE = "female"

class Customer(BaseSchema):
    customer_id: str
    name_surname: str
    gender: Gender
    age: int
    detailed_user_bio: str
    used_products: List[str]

class Customers(BaseSchema):
    company_name: str
    customers: List[Customer]

class CustomersList(BaseSchema):
    customers: List[Customers]