from datetime import date, time
from typing import Optional, List

from sqlalchemy import String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, relationship, mapped_column


class Base(DeclarativeBase):
    __abstract__ = True

    id: Mapped[int] = mapped_column(primary_key=True)

class Phone(Base):
    __tablename__ = "phone"

    price: Mapped[Optional[int]]
    hard_drive_storage_size_gb: Mapped[Optional[int]] = mapped_column()
    ram: Mapped[Optional[int]]
    battery_life: Mapped[Optional[time]]
    model: Mapped[Optional[str]] = mapped_column(String(100))
    manufacturer: Mapped[Optional[str]] = mapped_column(String(100))
    year_of_manufacture: Mapped[Optional[date]]
    rating: Mapped[Optional[float]]
    reviews: Mapped[List["Review"]] = relationship(
        back_populates="phone", cascade="all, delete-orphan"
    )

class Review(Base):
    __tablename__ = "review"

    text: Mapped[str] = mapped_column(Text())
    rating: Mapped[Optional[float]]
    author: Mapped[Optional[str]] = mapped_column(String(100))
    date: Mapped[Optional[date]]
    phone: Mapped["Phone"] = relationship(back_populates="review")

def init():
    engine = create_engine("sqlite:///database.db", echo=True)
    Base.metadata.create_all(engine)