import pandas as pd
from .dataset import MovieDataset



def start():

    print("\n")
    print("="*40)
    print(" Movie Recommendation AI ")
    print("="*40)


    dataset = MovieDataset(
        "movies.csv"
    )


    movies = dataset.load()


    print(
        f"\nLoaded {len(movies)} movies\n"
    )


    print("Choose some movies you like:\n")


    selected = dataset.get_random(5)


    for index, movie in enumerate(selected):

        print(
            index + 1,
            "-",
            movie.title,
            movie.genres
        )


    input(
        "\nPress Enter to continue..."
    )