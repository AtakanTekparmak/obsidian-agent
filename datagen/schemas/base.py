from datetime import datetime
from pydantic import BaseModel, ConfigDict

class BaseSchema(BaseModel):
    """Base schema for all schemas."""
    model_config = ConfigDict(json_encoders={datetime: lambda dt: dt.isoformat()}) 