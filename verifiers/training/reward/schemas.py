from typing import Optional

from pydantic import BaseModel

class JudgeResponse(BaseModel):
    total_num_of_facts: int
    num_of_facts_present: int
    ratio_of_facts_present: Optional[float] = None
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.total_num_of_facts > 0:
            self.ratio_of_facts_present = self.num_of_facts_present / self.total_num_of_facts
        else:
            self.ratio_of_facts_present = 0.0

