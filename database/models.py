import sqlalchemy
from sqlalchemy.orm import DeclarativeBase, Mapped
from sqlalchemy.testing.schema import mapped_column


class Model(DeclarativeBase):
    __tablename__ = "models"

    id: Mapped[int] = mapped_column(primary_key=True)