from pydantic import BaseModel, ConfigDict


class FrozenBaseModel(BaseModel):
    model_config = ConfigDict(frozen=True)
