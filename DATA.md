## Memory Agent Data Generation Pipeline

### Data Generation Steps

```
Personas -> Momentary Stories -> KB -> Multi-turn convo
                                  L -> QA 
```

#### Personas

```python
class Gender(str, enum.Enum):
    MALE = "male"
    FEMALE = "female"

class Relationship(BaseModel):
    name_surname: str
    relationship: str

class Birthplace(BaseModel):
    city: str
    country: str

class Persona(BaseModel):
    name_surname: str
    age: int
    gender: Gender
    birthplace: Birthplace
    occupation: str
    detailed_backstory: str
    relationships: List[Relationship]
```

#### Momentary Stories

