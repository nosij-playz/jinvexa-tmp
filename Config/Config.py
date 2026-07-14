import os

from dotenv import load_dotenv


class Config:

    def __init__(self):
        load_dotenv()

        self.model = os.getenv("MODEL", "gemma4:latest")

    def get_model(self):
        return self.model