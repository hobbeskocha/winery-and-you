from pydantic import BaseModel

class InputData(BaseModel):
    OrderVolume: int
    CustomerSegment: str 
    Division: str 
    SaleAmount: float