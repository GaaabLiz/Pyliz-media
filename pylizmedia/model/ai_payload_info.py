from dataclasses import dataclass

from pydantic import BaseModel


class AiPayloadMediaInfo2:

    def __init__(self):
        self.description = None
        self.tags = []
        self.filename = None
        self.text = []

    def __str__(self):
        return f"Description: {self.description}, Tags: {self.tags}, Filename: {self.filename}, Text: {self.text}"



class AiPayloadMediaInfo(BaseModel):
    description: str
    tags: list[str]
    filename: str
    text: list[str]

    def __str__(self):
        return f"Description: {self.description}, Tags: {self.tags}, Filename: {self.filename}, Text: {self.text}"

    class Config:
        extra = "ignore"

