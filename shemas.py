from datetime import time, date

from pydantic import BaseModel, Field

class Review(BaseModel):
    text: str = Field(...)
    rating: float = Field(gt=0, lt=5)
    author: str = Field()
    date: date = Field()
    phone: str = Field()

class Phone(BaseModel):
    price: int
    hard_drive_storage_size_gb: int = Field(gt=0)
    ram: int = Field(gt=0)
    battery_life: time
    model: str
    manufacturer: str
    year_of_manufacture: date
    rating: float = Field(gt=0, lt=5)
    reviews: set[Review]



