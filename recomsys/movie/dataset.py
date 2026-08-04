# movie/dataset.py

import os
import pandas as pd

from .models import Movie



class MovieDataset:

    def __init__(self, filename="movies.csv"):

        self.filename = filename
        self.movies = []



    def read_csv_auto(self, path):

        encodings = [
            "utf-8",
            "utf-8-sig",
            "latin1",
            "cp1252",
            "utf-16"
        ]


        for encoding in encodings:

            try:

                print(
                    f"Trying encoding: {encoding}"
                )

                return pd.read_csv(
                    path,
                    encoding=encoding
                )


            except UnicodeDecodeError:

                continue


        raise Exception(
            "Cannot detect CSV encoding"
        )



    def load(self):

        # مسیر پوشه movie
        folder = os.path.dirname(
            os.path.abspath(__file__)
        )


        full_path = os.path.join(
            folder,
            self.filename
        )


        if not os.path.exists(full_path):

            raise FileNotFoundError(
                f"Dataset not found:\n{full_path}"
            )


        data = self.read_csv_auto(
            full_path
        )


        required_columns = [

            "title",
            "genres",
            "runtime",
            "year"

        ]


        for column in required_columns:

            if column not in data.columns:

                raise Exception(
                    f"Missing column: {column}"
                )


        # حذف داده های ناقص

        data = data.dropna(
            subset=required_columns
        )


        self.movies.clear()


        for _, row in data.iterrows():

            try:

                movie = Movie(

                    title=str(
                        row["title"]
                    ),


           genres = str(row["genres"]).replace("-", "|").split("|"),


                    runtime=int(
                        row["runtime"]
                    ),


                    year=int(
                        row["year"]
                    )

                )


                self.movies.append(
                    movie
                )


            except Exception:

                # رد کردن ردیف خراب

                continue



        return self.movies



    def count(self):

        return len(
            self.movies
        )



    def get_random(self, count=20):

        import random


        if count > len(self.movies):

            count = len(self.movies)


        return random.sample(
            self.movies,
            count
        )