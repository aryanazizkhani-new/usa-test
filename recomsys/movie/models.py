from dataclasses import dataclass
import pandas as pd

@dataclass
class Movie:

    title: str
    genres: list
    runtime: int
    year: int