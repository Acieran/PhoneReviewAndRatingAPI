from datetime import time, date

from pydantic import BaseModel, Field


class Phone(BaseModel):

    price: int
    hard_drive_storage_size_gb: int = Field(gt=0)
    ram: int = Field(gt=0)
    battery_life: time
    model: str
    manufacturer: str
    year_of_manufacture: date
    rating: float = Field(gt=0)
    reviews: Mapped[List["Review"]] = relationship(
        back_populates="phone", cascade="all, delete-orphan"
    )
