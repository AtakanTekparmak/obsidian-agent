import enum
from typing import List

from data.schemas.base import BaseSchema


class Gender(str, enum.Enum):
    MALE = "male"
    FEMALE = "female"


class Relationship(BaseSchema):
    name_surname: str
    relationship: str


class Birthplace(BaseSchema):
    city: str
    country: str


class Persona(BaseSchema):
    name_surname: str
    age: int
    gender: Gender
    birthplace: Birthplace
    occupation: str
    detailed_backstory: str
    relationships: List[Relationship]


class Personas(BaseSchema):
    personas: List[Persona]
