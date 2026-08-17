# -*- coding: utf-8 -*-

"""
===============================================================================
DECISION SYSTEM v26 FIXED
===============================================================================

V26 LOGIC:
    Manual TF-IDF + SVD + Decision Tree + Small MLP + META
    Independent Gate
    Leakage-safe OOF
    No .transform()
    CPU THREAD LIMIT = 2
    MAX ROWS / DOMAIN = 5000

FIXES:
    1. Correct artist metadata join.
    2. Robust artist ID/name matching.
    3. Better multi-artist graph matching.
    4. Safer TF-IDF artifact serialization.
    5. Safer tiny-dataset handling.
    6. Safer MLP training.
    7. Safer META training.
    8. Safer OOF generation.
    9. Correct feature-schema alignment.
   10. Better numeric parsing.
   11. Better release-date parsing.
   12. No sklearn .transform().
   13. No data leakage from test set.
   14. Independent gate remains independent.
   15. Existing V26 architecture preserved.

SONG SOURCE:
    database/tracks.csv
    database/artists.csv
    database/dict_artists.json

SONG TARGET:
    tracks.csv -> popularity / Popularity

OTHER DOMAINS:
    APP
    GAME
    BOOK
    MOVIE
"""

# =============================================================================
# CPU
# =============================================================================

import os

CPU_THREADS = 2

os.environ["OMP_NUM_THREADS"] = str(CPU_THREADS)
os.environ["MKL_NUM_THREADS"] = str(CPU_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(CPU_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(CPU_THREADS)


# =============================================================================
# IMPORTS
# =============================================================================

import gc
import json
import math
import random
import re
import time

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor


# =============================================================================
# CONFIG
# =============================================================================

SEED = 42

MAX_ROWS = 5000

OUTER_TEST_SIZE = 0.20
GATE_SIZE = 0.20

OOF_FOLDS = 3

TOP_K = 30

TFIDF_MAX_FEATURES = 4000
SONG_TFIDF_MAX_FEATURES = 5000

SVD_COMPONENTS = 24

USE_MLP = True

MLP_DOMINANCE_FACTOR = 1.10

SONG_POPULARITY_TARGET = "Popularity"

random.seed(SEED)
np.random.seed(SEED)


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

DB = BASE_DIR / "database"

OUTPUT = BASE_DIR / "decision_system_v26_output"

OUTPUT.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# DATASETS
# =============================================================================

DATASETS = {
    "song": DB / "tracks.csv",
    "song_artists": DB / "artists.csv",
    "song_graph": DB / "dict_artists.json",

    "book": DB / "books_1.Best_Books_Ever.csv",
    "app": DB / "googleplaystore.csv",
    "game": DB / "computer_games.csv",
    "movie": DB / "TMDB_movie_dataset_v11.csv",
}


# =============================================================================
# DOMAIN CONFIG
# =============================================================================

@dataclass
class DomainConfig:
    target: Optional[str]
    text: list


CFG = {

    "song": DomainConfig(
        target="Popularity",
        text=[
            "name",
            "track_name",
            "song",
            "artists",
            "artist",
            "album",
            "genre",
        ],
    ),

    "book": DomainConfig(
        target="rating",
        text=[
            "title",
            "author",
            "description",
            "genres",
            "characters",
            "awards",
            "setting",
            "series",
            "publisher",
        ],
    ),

    "app": DomainConfig(
        target="Rating",
        text=[
            "App",
            "Category",
            "Type",
            "Content Rating",
            "Genres",
            "Current Ver",
            "Android Ver",
        ],
    ),

    "game": DomainConfig(
        target=None,
        text=[
            "Name",
            "Developer",
            "Producer",
            "Genre",
            "Operating System",
        ],
    ),

    "movie": DomainConfig(
        target="vote_average",
        text=[
            "title",
            "status",
            "original_language",
            "original_title",
            "overview",
            "tagline",
            "genres",
            "production_companies",
            "production_countries",
            "spoken_languages",
            "keywords",
        ],
    ),
}


# =============================================================================
# MISSING
# =============================================================================

MISSING = {
    "",
    "nan",
    "none",
    "null",
    "n/a",
    "na",
    "unknown",
    "-",
    "--",
}


# =============================================================================
# TREE
# =============================================================================

TREE_SETTINGS = {
    "song": {
        "max_depth": 11,
        "min_samples_leaf": 8,
        "min_samples_split": 16,
    },

    "book": {
        "max_depth": 11,
        "min_samples_leaf": 8,
        "min_samples_split": 16,
    },

    "app": {
        "max_depth": 11,
        "min_samples_leaf": 8,
        "min_samples_split": 16,
    },

    "game": {
        "max_depth": 11,
        "min_samples_leaf": 8,
        "min_samples_split": 16,
    },

    "movie": {
        "max_depth": 11,
        "min_samples_leaf": 8,
        "min_samples_split": 16,
    },
}


# =============================================================================
# MLP
# =============================================================================

MLP_SETTINGS = {
    "song": {
        "hidden": (64, 32),
        "alpha": 0.006,
        "max_iter": 80,
    },

    "book": {
        "hidden": (64, 32),
        "alpha": 0.006,
        "max_iter": 80,
    },

    "app": {
        "hidden": (64, 32),
        "alpha": 0.006,
        "max_iter": 80,
    },

    "game": {
        "hidden": (64, 32),
        "alpha": 0.006,
        "max_iter": 80,
    },

    "movie": {
        "hidden": (64, 32),
        "alpha": 0.006,
        "max_iter": 80,
    },
}


# =============================================================================
# NORMALIZATION
# =============================================================================

def normalize_column(value):

    value = str(value).strip().lower()

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value


def resolve_column(dataframe, wanted):

    wanted = normalize_column(wanted)

    for column in dataframe.columns:

        if normalize_column(column) == wanted:
            return column

    return None


def resolve_first_column(dataframe, candidates):

    normalized = {
        normalize_column(c): c
        for c in dataframe.columns
    }

    for candidate in candidates:

        key = normalize_column(candidate)

        if key in normalized:
            return normalized[key]

    return None


# =============================================================================
# TEXT
# =============================================================================

def clean_text(value):

    if pd.isna(value):
        return ""

    value = str(value).strip()

    if value.lower() in MISSING:
        return ""

    value = (
        value
        .replace("\n", " ")
        .replace("\r", " ")
        .replace("\t", " ")
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip()


def text_series(dataframe, requested):

    column = resolve_column(
        dataframe,
        requested,
    )

    if column is None:

        return pd.Series(
            [""] * len(dataframe),
            index=dataframe.index,
            dtype="object",
        )

    return (
        dataframe[column]
        .fillna("")
        .astype(str)
        .map(clean_text)
    )


# =============================================================================
# NUMBERS
# =============================================================================

def parse_number_value(value):

    if pd.isna(value):
        return np.nan

    if isinstance(
        value,
        (
            int,
            float,
            np.integer,
            np.floating,
        ),
    ):

        value = float(value)

        if np.isfinite(value):
            return value

        return np.nan

    text = str(value).strip().lower()

    if text in MISSING:
        return np.nan

    text = (
        text
        .replace(",", "")
        .replace("$", "")
        .replace("€", "")
        .replace("£", "")
        .replace("+", "")
    )

    multiplier = 1.0

    if "billion" in text:
        multiplier = 1e9

    elif "million" in text:
        multiplier = 1e6

    elif "thousand" in text:
        multiplier = 1e3

    elif text.endswith("bn"):
        multiplier = 1e9

    elif text.endswith("m"):
        multiplier = 1e6

    elif text.endswith("k"):
        multiplier = 1e3

    match = re.search(
        r"[-+]?\d+(?:\.\d+)?",
        text,
    )

    if match is None:
        return np.nan

    try:

        return (
            float(match.group())
            *
            multiplier
        )

    except Exception:

        return np.nan


def number_series(dataframe, requested):

    column = resolve_column(
        dataframe,
        requested,
    )

    if column is None:

        return pd.Series(
            np.zeros(
                len(dataframe),
                dtype=np.float32,
            ),
            index=dataframe.index,
        )

    return (
        dataframe[column]
        .map(parse_number_value)
        .fillna(0)
        .astype(np.float32)
    )


# =============================================================================
# FEATURES
# =============================================================================

def word_count(series):

    return (
        series
        .astype(str)
        .str.findall(r"\b[\w']+\b")
        .str.len()
        .fillna(0)
        .astype(np.float32)
    )


def separator_count(series):

    return (
        series
        .astype(str)
        .str.count(r",|;|\||/|&")
        .fillna(0)
        .add(1)
        .astype(np.float32)
    )


def safe_ratio(numerator, denominator):

    numerator = np.asarray(
        numerator,
        dtype=np.float64,
    )

    denominator = np.asarray(
        denominator,
        dtype=np.float64,
    )

    result = (
        numerator
        /
        np.maximum(
            denominator,
            1.0,
        )
    )

    return np.nan_to_num(
        result,
        nan=0,
        posinf=0,
        neginf=0,
    ).astype(np.float32)


def add_feature(
    arrays,
    names,
    values,
    name,
    upper=None,
):

    values = np.asarray(
        values,
        dtype=np.float32,
    )

    if upper is not None:

        values = np.clip(
            values,
            0,
            float(upper),
        )

    arrays.append(values)
    names.append(name)


# =============================================================================
# SAFE SPLIT
# =============================================================================

def split_indices(values, test_size, seed):

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    n = len(values)

    if n < 2:

        raise ValueError(
            "At least 2 samples are required."
        )

    indices = np.arange(n)

    test_count = int(
        round(
            n * float(test_size)
        )
    )

    test_count = max(
        1,
        min(
            n - 1,
            test_count,
        ),
    )

    # -------------------------------------------------------------------------
    # Try stratification by quantile bins.
    # -------------------------------------------------------------------------

    try:

        unique_count = np.unique(values).size

        if unique_count >= 10:

            bin_count = min(
                10,
                max(
                    2,
                    n // 50,
                ),
            )

            bins = pd.qcut(
                values,
                q=bin_count,
                labels=False,
                duplicates="drop",
            )

            bins = np.asarray(bins)

            if not np.any(pd.isna(bins)):

                bins = bins.astype(np.int64)

                counts = np.bincount(bins)

                if (
                    len(counts) > 1
                    and
                    np.min(counts) >= 2
                ):

                    return train_test_split(
                        indices,
                        test_size=test_count,
                        random_state=seed,
                        shuffle=True,
                        stratify=bins,
                    )

    except Exception:
        pass

    rng = np.random.default_rng(seed)

    shuffled = indices.copy()

    rng.shuffle(shuffled)

    return (
        shuffled[test_count:],
        shuffled[:test_count],
    )


# =============================================================================
# TARGET SCALER
# =============================================================================

class TargetScaler:

    def fit(self, values):

        values = np.asarray(
            values,
            dtype=np.float64,
        )

        if len(values) == 0:

            self.minimum = 0.0
            self.maximum = 1.0

            return self

        self.minimum = float(
            np.min(values)
        )

        self.maximum = float(
            np.max(values)
        )

        if (
            not np.isfinite(self.minimum)
            or
            not np.isfinite(self.maximum)
        ):

            self.minimum = 0.0
            self.maximum = 1.0

        if self.maximum <= self.minimum:

            self.maximum = (
                self.minimum
                +
                1.0
            )

        return self

    def normalize(self, values):

        values = np.asarray(
            values,
            dtype=np.float64,
        )

        result = (
            values
            -
            self.minimum
        ) / (
            self.maximum
            -
            self.minimum
        )

        return np.clip(
            result,
            0,
            1,
        ).astype(np.float32)

    def denormalize(self, values):

        values = np.asarray(
            values,
            dtype=np.float64,
        )

        return (
            values
            *
            (
                self.maximum
                -
                self.minimum
            )
            +
            self.minimum
        )


# =============================================================================
# SONG MATCH KEY
# =============================================================================

def normalize_match_key(value):

    value = clean_text(value).lower()

    value = value.replace(
        "&",
        "and",
    )

    value = re.sub(
        r"[^\w\s]",
        " ",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip()


def normalize_id_key(value):

    # -----------------------------------------------------------------------
    # Spotify-style artist IDs are case-sensitive base62 strings. Unlike
    # artist NAMES, IDs must never be lowercased or otherwise mangled, or
    # two distinct artists whose IDs differ only in case would collide.
    # -----------------------------------------------------------------------

    value = clean_text(value)

    return value.strip()


def split_artist_names(value):

    value = clean_text(value)

    if not value:
        return []

    parts = re.split(
        r"\s*(?:,|;|\||/|&|\bx\b|\bfeat\.?\b|\bfeaturing\b)\s*",
        value,
        flags=re.IGNORECASE,
    )

    result = []

    for part in parts:

        key = normalize_match_key(part)

        if key:
            result.append(key)

    # Also include the complete artist field.
    full_key = normalize_match_key(value)

    if full_key:
        result.append(full_key)

    # Preserve order and remove duplicates.
    result = list(
        dict.fromkeys(result)
    )

    return result


def extract_artist_ids(raw_id_value):

    # -------------------------------------------------------------------------
    # Pull individual Spotify-style artist IDs out of a track's raw
    # id_artists field, e.g. "['1234abcd', '5678efgh']". IDs are
    # case-sensitive base62 strings, so normalize_id_key (which preserves
    # case) is used instead of normalize_match_key.
    # -------------------------------------------------------------------------

    id_text = clean_text(raw_id_value)

    if not id_text:
        return []

    ids = re.findall(
        r"[A-Za-z0-9]{5,}",
        id_text,
    )

    result = [
        normalize_id_key(x)
        for x in ids
        if normalize_id_key(x)
    ]

    return list(
        dict.fromkeys(result)
    )


# =============================================================================
# ARTIST GRAPH
# =============================================================================

def load_artist_graph(path):

    if not path.exists():

        print(
            "Artist graph not found."
        )

        return {}

    print(
        "Loading artist relation graph:",
        path,
    )

    try:

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:

            data = json.load(f)

    except Exception as error:

        print(
            "Artist graph load failed:",
            repr(error),
        )

        return {}

    graph = {}

    # -------------------------------------------------------------------------
    # IMPORTANT FIX:
    #
    # dict_artists.json (as shipped with this tracks/artists dataset) is
    # keyed by Spotify ARTIST ID, with values being lists of related
    # artist IDs -- not artist names. IDs are case-sensitive base62
    # strings, so building the graph with normalize_match_key (which
    # lowercases and strips punctuation, intended for names) silently
    # corrupted every key. Combined with the fact that lookups were also
    # being done by artist NAME rather than ID, this made every graph
    # lookup miss (0 matches, no matter how large the graph was).
    #
    # normalize_id_key preserves case and only trims whitespace, which is
    # correct for ID-keyed data. See calculate_artist_graph_features()
    # below for the matching lookup-side fix.
    # -------------------------------------------------------------------------

    if isinstance(data, dict):

        for key, value in data.items():

            key = normalize_id_key(key)

            if not key:
                continue

            neighbors = []

            if isinstance(value, dict):

                neighbors = list(
                    value.keys()
                )

            elif isinstance(value, list):

                neighbors = value

            elif isinstance(value, str):

                neighbors = [
                    value
                ]

            graph.setdefault(
                key,
                set(),
            )

            for neighbor in neighbors:

                neighbor_key = (
                    normalize_id_key(
                        neighbor
                    )
                )

                if neighbor_key:

                    graph[key].add(
                        neighbor_key
                    )

    elif isinstance(data, list):

        for item in data:

            if isinstance(item, dict):

                a = (
                    item.get("artist")
                    or
                    item.get("source")
                    or
                    item.get("from")
                    or
                    item.get("artist1")
                )

                b = (
                    item.get("related")
                    or
                    item.get("target")
                    or
                    item.get("to")
                    or
                    item.get("artist2")
                )

                if a is not None and b is not None:

                    a = normalize_id_key(a)
                    b = normalize_id_key(b)

                    if a and b:

                        graph.setdefault(
                            a,
                            set(),
                        ).add(b)

                        graph.setdefault(
                            b,
                            set(),
                        ).add(a)

            elif (
                isinstance(
                    item,
                    (list, tuple),
                )
                and
                len(item) >= 2
            ):

                a = normalize_id_key(
                    item[0]
                )

                b = normalize_id_key(
                    item[1]
                )

                if a and b:

                    graph.setdefault(
                        a,
                        set(),
                    ).add(b)

                    graph.setdefault(
                        b,
                        set(),
                    ).add(a)

    clean_graph = {}

    for key, values in graph.items():

        clean_graph[key] = set(
            normalize_id_key(x)
            for x in values
            if normalize_id_key(x)
        )

    print(
        "Artist graph nodes:",
        len(clean_graph),
    )

    return clean_graph


def calculate_artist_graph_features(
    id_keys,
    name_keys,
    graph,
):

    # -------------------------------------------------------------------------
    # IMPORTANT FIX:
    #
    # Previously this function only ever received name-derived keys
    # (from split_artist_names on the track's artist-name text) and
    # looked those up in a graph that is actually keyed by artist ID.
    # That guaranteed zero matches.
    #
    # Now we try ID-based candidates first (the correct key space for
    # dict_artists.json), and fall back to name-based candidates so
    # datasets where the graph genuinely is name-keyed still work.
    # -------------------------------------------------------------------------

    keys = list(
        dict.fromkeys(
            list(id_keys or [])
            +
            list(name_keys or [])
        )
    )

    if not keys:

        return (
            0.0,
            0.0,
            0.0,
        )

    all_neighbors = set()
    exists = 0.0

    for key in keys:

        neighbors = graph.get(
            key,
            set(),
        )

        if neighbors:

            exists = 1.0

            all_neighbors.update(
                neighbors
            )

    if not all_neighbors:

        return (
            0.0,
            0.0,
            exists,
        )

    degree = min(
        len(all_neighbors),
        5000,
    )

    second_degree = 0

    for neighbor in list(
        all_neighbors
    )[:100]:

        second_degree += len(
            graph.get(
                neighbor,
                set(),
            )
        )

    second_degree = min(
        second_degree,
        10000,
    )

    return (
        float(degree),
        float(second_degree),
        float(exists),
    )


# =============================================================================
# ARTIST LOOKUP
# =============================================================================

def build_artist_lookup(
    artists,
    artist_name_col,
    artist_id_col,
    followers_col,
    popularity_col,
    genres_col,
):

    lookup = {}

    if artists is None or len(artists) == 0:
        return lookup

    for _, row in artists.iterrows():

        if artist_id_col is not None:

            artist_id = normalize_id_key(
                row.get(
                    artist_id_col,
                    "",
                )
            )

            if artist_id:

                lookup[
                    ("id", artist_id)
                ] = (
                    parse_number_value(
                        row.get(
                            followers_col,
                            0,
                        )
                    )
                    if followers_col
                    else 0.0,

                    parse_number_value(
                        row.get(
                            popularity_col,
                            0,
                        )
                    )
                    if popularity_col
                    else 0.0,

                    clean_text(
                        row.get(
                            genres_col,
                            "",
                        )
                    )
                    if genres_col
                    else "",
                )

        if artist_name_col is not None:

            artist_name = normalize_match_key(
                row.get(
                    artist_name_col,
                    "",
                )
            )

            if artist_name:

                key = (
                    "name",
                    artist_name,
                )

                if key not in lookup:

                    lookup[key] = (
                        parse_number_value(
                            row.get(
                                followers_col,
                                0,
                            )
                        )
                        if followers_col
                        else 0.0,

                        parse_number_value(
                            row.get(
                                popularity_col,
                                0,
                            )
                        )
                        if popularity_col
                        else 0.0,

                        clean_text(
                            row.get(
                                genres_col,
                                "",
                            )
                        )
                        if genres_col
                        else "",
                    )

    return lookup


# =============================================================================
# SONG LOADER
# =============================================================================

def load_song_dataset():

    tracks_path = DATASETS["song"]
    artists_path = DATASETS["song_artists"]
    graph_path = DATASETS["song_graph"]

    if not tracks_path.exists():

        raise FileNotFoundError(
            "Missing tracks.csv:\n"
            +
            str(tracks_path)
        )

    print()
    print(
        "TRACKS DATASET:",
        tracks_path,
    )

    tracks = pd.read_csv(
        tracks_path,
        low_memory=False,
    )

    print(
        "TRACKS SHAPE:",
        tracks.shape,
    )

    # =========================================================================
    # EARLY ROW CAP
    #
    # IMPORTANT FIX:
    #
    # Previously the MAX_ROWS cap was only applied inside train_domain(),
    # which runs AFTER load_song_dataset() has already:
    #   - iterated every row for artist metadata matching, and
    #   - iterated every row for artist graph feature calculation.
    #
    # On a full tracks.csv (hundreds of thousands of rows) that means doing
    # two expensive per-row Python loops over data that gets thrown away a
    # few steps later anyway. Sampling down to MAX_ROWS here, right after
    # reading the CSV, means all the heavy per-row work below only ever
    # touches the rows we will actually train on.
    # =========================================================================

    if len(tracks) > MAX_ROWS:

        tracks = (
            tracks
            .sample(
                n=MAX_ROWS,
                random_state=SEED,
            )
            .reset_index(drop=True)
        )

        print(
            "TRACKS SAMPLED TO MAX_ROWS:",
            tracks.shape,
        )

    # =========================================================================
    # ARTISTS
    # =========================================================================

    artists = None

    if artists_path.exists():

        print(
            "ARTISTS DATASET:",
            artists_path,
        )

        artists = pd.read_csv(
            artists_path,
            low_memory=False,
        )

        print(
            "ARTISTS SHAPE:",
            artists.shape,
        )

    else:

        print(
            "artists.csv not found."
        )

    # =========================================================================
    # GRAPH
    # =========================================================================

    graph = load_artist_graph(
        graph_path
    )

    # =========================================================================
    # RESOLVE TRACK COLUMNS
    # =========================================================================

    track_artist_col = resolve_first_column(
        tracks,
        [
            "artists",
            "artist",
            "Artist",
            "artist_name",
        ],
    )

    track_artist_id_col = resolve_first_column(
        tracks,
        [
            "id_artists",
            "artist_ids",
            "artist_id",
            "artists_id",
        ],
    )

    track_name_col = resolve_first_column(
        tracks,
        [
            "name",
            "track_name",
            "song",
            "title",
            "Title",
        ],
    )

    album_col = resolve_first_column(
        tracks,
        [
            "album",
            "Album",
            "album_name",
        ],
    )

    if track_artist_col is None:

        tracks["artist"] = ""

        track_artist_col = "artist"

    if track_name_col is None:

        tracks["song"] = ""

        track_name_col = "song"

    # =========================================================================
    # STANDARD ALIASES
    # =========================================================================

    tracks["artist"] = (
        tracks[track_artist_col]
        .fillna("")
        .astype(str)
        .map(clean_text)
    )

    tracks["song"] = (
        tracks[track_name_col]
        .fillna("")
        .astype(str)
        .map(clean_text)
    )

    if album_col is not None:

        tracks["album"] = (
            tracks[album_col]
            .fillna("")
            .astype(str)
            .map(clean_text)
        )

    else:

        tracks["album"] = ""

    # =========================================================================
    # ARTIST METADATA
    #
    # IMPORTANT FIX:
    #
    # The previous code initialized:
    #
    #     tracks["artist_metadata_followers"] = 0
    #
    # and then merged a dataframe containing the same column name.
    #
    # pandas could therefore create:
    #
    #     artist_metadata_followers
    #     artist_metadata_followers_artist_joined
    #
    # while the original zero column remained active.
    #
    # This version performs the lookup explicitly and writes directly into
    # the final columns.
    # =========================================================================

    tracks["artist_metadata_followers"] = (
        np.zeros(
            len(tracks),
            dtype=np.float32,
        )
    )

    tracks["artist_metadata_popularity"] = (
        np.zeros(
            len(tracks),
            dtype=np.float32,
        )
    )

    tracks["artist_metadata_genres"] = ""

    if artists is not None and len(artists):

        artist_name_col = resolve_first_column(
            artists,
            [
                "name",
                "artist",
                "artist_name",
                "Artist",
            ],
        )

        artist_id_col = resolve_first_column(
            artists,
            [
                "id",
                "artist_id",
                "id_artist",
            ],
        )

        followers_col = resolve_first_column(
            artists,
            [
                "followers",
                "artist_followers",
            ],
        )

        popularity_col = resolve_first_column(
            artists,
            [
                "popularity",
                "artist_popularity",
            ],
        )

        genres_col = resolve_first_column(
            artists,
            [
                "genres",
                "genre",
                "Genres",
            ],
        )

        artist_lookup = build_artist_lookup(
            artists,
            artist_name_col,
            artist_id_col,
            followers_col,
            popularity_col,
            genres_col,
        )

        followers_values = np.zeros(
            len(tracks),
            dtype=np.float32,
        )

        popularity_values = np.zeros(
            len(tracks),
            dtype=np.float32,
        )

        genre_values = np.empty(
            len(tracks),
            dtype=object,
        )

        genre_values[:] = ""

        matched = 0

        # ---------------------------------------------------------------------
        # Prefer ID matching.
        # ---------------------------------------------------------------------

        for i, row in tracks.iterrows():

            found = None

            if (
                track_artist_id_col is not None
                and
                artist_id_col is not None
            ):

                raw_id = row.get(
                    track_artist_id_col,
                    "",
                )

                # Spotify-style id_artists may contain:
                # ['id1','id2']
                id_text = clean_text(
                    raw_id
                )

                ids = re.findall(
                    r"[A-Za-z0-9]{5,}",
                    id_text,
                )

                if ids:

                    # ---------------------------------------------------------
                    # FIX: IDs are case-sensitive. Use normalize_id_key
                    # (no lowercasing) instead of normalize_match_key, or
                    # distinct artists whose IDs differ only by case would
                    # incorrectly collide onto the same lookup key.
                    # ---------------------------------------------------------

                    candidates = [
                        normalize_id_key(x)
                        for x in ids
                    ]

                    candidates.append(
                        normalize_id_key(
                            id_text
                        )
                    )

                    for candidate in candidates:

                        if candidate:

                            found = artist_lookup.get(
                                (
                                    "id",
                                    candidate,
                                )
                            )

                            if found is not None:
                                break

            # -----------------------------------------------------------------
            # Name fallback.
            # -----------------------------------------------------------------

            if found is None:

                artist_text = row.get(
                    "artist",
                    "",
                )

                artist_keys = (
                    split_artist_names(
                        artist_text
                    )
                )

                values = []

                for key in artist_keys:

                    item = artist_lookup.get(
                        (
                            "name",
                            key,
                        )
                    )

                    if item is not None:

                        values.append(item)

                if values:

                    followers = max(
                        float(x[0] or 0)
                        for x in values
                    )

                    popularity = max(
                        float(x[1] or 0)
                        for x in values
                    )

                    genres = []

                    for x in values:

                        if x[2]:

                            genres.append(
                                x[2]
                            )

                    found = (
                        followers,
                        popularity,
                        " ".join(
                            genres
                        ),
                    )

            if found is not None:

                matched += 1

                followers_values[i] = (
                    0.0
                    if not np.isfinite(
                        float(found[0] or 0)
                    )
                    else float(found[0] or 0)
                )

                popularity_values[i] = (
                    0.0
                    if not np.isfinite(
                        float(found[1] or 0)
                    )
                    else float(found[1] or 0)
                )

                genre_values[i] = clean_text(
                    found[2]
                )

        tracks[
            "artist_metadata_followers"
        ] = followers_values

        tracks[
            "artist_metadata_popularity"
        ] = popularity_values

        tracks[
            "artist_metadata_genres"
        ] = pd.Series(
            genre_values,
            index=tracks.index,
        ).fillna("").astype(str)

        print(
            "Artist metadata matched:",
            matched,
            "/",
            len(tracks),
            "(",
            round(
                100.0 * matched / max(
                    1,
                    len(tracks),
                ),
                2,
            ),
            "%)"
        )

    # =========================================================================
    # ARTIST GRAPH
    # =========================================================================

    graph_degree = np.zeros(
        len(tracks),
        dtype=np.float32,
    )

    graph_second_degree = np.zeros(
        len(tracks),
        dtype=np.float32,
    )

    graph_exists = np.zeros(
        len(tracks),
        dtype=np.float32,
    )

    # -------------------------------------------------------------------------
    # IMPORTANT FIX:
    #
    # This used to pass only the artist NAME text to
    # calculate_artist_graph_features(), which then looked it up in a
    # graph keyed by artist ID -- guaranteeing 0 matches regardless of
    # graph size. We now extract each track's artist ID(s) from the same
    # id column used for artist-metadata matching above (when available)
    # and pass those as the primary lookup keys, with name-derived keys
    # as a fallback.
    # -------------------------------------------------------------------------

    artist_id_values = (
        tracks[track_artist_id_col]
        if track_artist_id_col is not None
        else None
    )

    for i in range(len(tracks)):

        id_keys = (
            extract_artist_ids(
                artist_id_values.iat[i]
            )
            if artist_id_values is not None
            else []
        )

        name_keys = split_artist_names(
            tracks["artist"].iat[i]
        )

        degree, second, exists = (
            calculate_artist_graph_features(
                id_keys,
                name_keys,
                graph,
            )
        )

        graph_degree[i] = degree
        graph_second_degree[i] = second
        graph_exists[i] = exists

    tracks[
        "artist_graph_degree"
    ] = graph_degree

    tracks[
        "artist_graph_second_degree"
    ] = graph_second_degree

    tracks[
        "artist_graph_exists"
    ] = graph_exists

    print(
        "Artist graph matched:",
        int(
            np.sum(
                graph_exists > 0
            )
        ),
        "/",
        len(tracks),
    )

    # =========================================================================
    # RELEASE DATE
    # =========================================================================

    release_col = resolve_first_column(
        tracks,
        [
            "release_date",
            "Release Date",
            "release",
            "date",
        ],
    )

    if release_col is not None:

        tracks[
            "release_date_normalized"
        ] = (
            tracks[release_col]
            .fillna("")
            .astype(str)
        )

    else:

        tracks[
            "release_date_normalized"
        ] = ""

    # =========================================================================
    # GENRE
    # =========================================================================

    tracks["genre"] = (
        tracks[
            "artist_metadata_genres"
        ]
        .fillna("")
        .astype(str)
    )

    # =========================================================================
    # TARGET
    # =========================================================================

    popularity_col = resolve_first_column(
        tracks,
        [
            "popularity",
            "Popularity",
            "spotify_popularity",
        ],
    )

    if popularity_col is None:

        raise ValueError(
            "tracks.csv does not contain "
            "popularity / Popularity."
        )

    tracks["Popularity"] = (
        tracks[popularity_col]
        .map(parse_number_value)
    )

    # =========================================================================
    # CLEAN
    # =========================================================================

    for column in [
        "_artist_join_id",
        "_artist_join_name",
    ]:

        if column in tracks.columns:

            tracks.drop(
                columns=[column],
                inplace=True,
                errors="ignore",
            )

    del artists

    gc.collect()

    print(
        "Loaded:",
        len(tracks),
        "rows |",
        len(tracks.columns),
        "columns",
    )

    return (
        tracks,
        tracks_path,
    )


# =============================================================================
# BUILD TEXT
# =============================================================================

def build_text(dataframe, category):

    result = pd.Series(
        [""] * len(dataframe),
        index=dataframe.index,
        dtype="object",
    )

    for requested in CFG[
        category
    ].text:

        values = text_series(
            dataframe,
            requested,
        )

        result = (
            result
            +
            " "
            +
            values
        )

    if category == "song":

        metadata_genres = text_series(
            dataframe,
            "artist_metadata_genres",
        )

        result = (
            result
            +
            " "
            +
            metadata_genres
        )

        release_text = text_series(
            dataframe,
            "release_date_normalized",
        )

        result = (
            result
            +
            " "
            +
            release_text
        )

    result = (
        result
        .str.replace(
            r"\s+",
            " ",
            regex=True,
        )
        .str.strip()
    )

    result = result.mask(
        result.eq(""),
        "unknown",
    )

    return result


# =============================================================================
# TARGET
# =============================================================================

def build_target(dataframe, category):

    if category == "game":

        name = text_series(
            dataframe,
            "Name",
        )

        developer = text_series(
            dataframe,
            "Developer",
        )

        genre = text_series(
            dataframe,
            "Genre",
        )

        date_column = resolve_column(
            dataframe,
            "Date Released",
        )

        if date_column:

            dates = pd.to_datetime(
                dataframe[date_column],
                errors="coerce",
            )

            years = (
                dates.dt.year
                .fillna(2000)
                .values
                .astype(np.float32)
            )

        else:

            years = np.full(
                len(dataframe),
                2000,
                dtype=np.float32,
            )

        genre_count = separator_count(
            genre
        ).values

        name_length = (
            name.str.len()
            .values
            .astype(np.float32)
        )

        developer_length = (
            developer.str.len()
            .values
            .astype(np.float32)
        )

        def norm(values):

            values = np.asarray(
                values,
                dtype=np.float64,
            )

            low = np.min(values)
            high = np.max(values)

            if high <= low:

                return np.zeros(
                    len(values),
                    dtype=np.float32,
                )

            return (
                (values - low)
                /
                (high - low)
            ).astype(np.float32)

        score = (
            0.50 * norm(years)
            +
            0.20 *
            (
                name_length > 0
            ).astype(np.float32)
            +
            0.15 * norm(genre_count)
            +
            0.10 * norm(name_length)
            +
            0.05 * norm(developer_length)
        )

        return (
            np.clip(
                score,
                0,
                1,
            ).astype(np.float32),

            np.ones(
                len(dataframe),
                dtype=bool,
            ),

            "PSEUDO:game_score",
        )

    target_column = resolve_column(
        dataframe,
        CFG[category].target,
    )

    if target_column is None:

        raise ValueError(
            "Missing target column "
            +
            str(
                CFG[category].target
            )
        )

    values = (
        dataframe[target_column]
        .map(parse_number_value)
        .astype(float)
        .values
    )

    if category == "song":

        low = 0.0
        high = 100.0

    elif category in (
        "book",
        "app",
    ):

        low = 0.0
        high = 5.0

    else:

        low = 0.0
        high = 10.0

    valid = (
        np.isfinite(values)
        &
        (values >= low)
        &
        (values <= high)
    )

    return (
        np.nan_to_num(
            values,
            nan=0,
        ).astype(np.float32),

        valid,

        "REAL:" + str(
            CFG[category].target
        ),
    )


# =============================================================================
# STRUCTURED FEATURES
# =============================================================================

# =============================================================================
# STRUCTURED FEATURES
# =============================================================================

def make_features(dataframe, category):

    arrays = []
    names = []

    # =========================================================================
    # SONG
    # =========================================================================

    if category == "song":

        artist = text_series(dataframe, "artist")
        song = text_series(dataframe, "song")
        album = text_series(dataframe, "album")
        genre = text_series(dataframe, "genre")

        # --- Advanced Text & Structural Features ---
        num_artists = (
            artist.str.count(r"(?:,|&|\bfeat\.?\b|\bfeaturing\b|\bx\b)") + 1
        ).fillna(1).astype(np.float32).clip(upper=20).values

        has_feat = (
            artist.str.contains(r"\b(?:feat\.?|featuring|ft\.?|x)\b", case=False, na=False)
            .astype(np.float32)
            .values
        )

        is_remix = (
            song.str.contains(r"\b(?:remix|edit|mix|version)\b", case=False, na=False)
            .astype(np.float32)
            .values
        )

        is_acoustic_live = (
            song.str.contains(r"\b(?:acoustic|unplugged|live)\b", case=False, na=False)
            .astype(np.float32)
            .values
        )

        features = [
            (artist.str.len().values, "artist_length", 300),
            (song.str.len().values, "song_length", 300),
            (album.str.len().values, "album_length", 300),
            (word_count(artist), "artist_words", 50),
            (word_count(song), "song_words", 50),
            (word_count(album), "album_words", 50),
            (separator_count(genre), "genre_count", 50),
            (num_artists, "num_artists", 20),
            (has_feat, "has_feat", 1),
            (is_remix, "is_remix", 1),
            (is_acoustic_live, "is_acoustic_live", 1),
        ]

        for value, name, maximum in features:
            add_feature(arrays, names, value, name, maximum)

        # --- Audio Features & Derived Metrics ---
        danceability = number_series(dataframe, "danceability").values
        energy = number_series(dataframe, "energy").values
        valence = number_series(dataframe, "valence").values
        acousticness = number_series(dataframe, "acousticness").values
        instrumentalness = number_series(dataframe, "instrumentalness").values
        liveness = number_series(dataframe, "liveness").values
        speechiness = number_series(dataframe, "speechiness").values
        loudness = number_series(dataframe, "loudness").values
        tempo = number_series(dataframe, "tempo").values
        duration_ms = number_series(dataframe, "duration_ms").values

        # Derived audio features (interaction terms are highly predictive for music)
        energy_valence_ratio = safe_ratio(energy, valence + 0.1)
        acoustic_energy = (acousticness * energy).astype(np.float32)
        dance_energy = (danceability * energy).astype(np.float32)
        duration_minutes = (duration_ms / 60000.0).astype(np.float32)
        is_long_track = (duration_ms > 300000).astype(np.float32)  # > 5 minutes

        add_feature(arrays, names, energy_valence_ratio, "energy_valence_ratio", 10)
        add_feature(arrays, names, acoustic_energy, "acoustic_energy", 1)
        add_feature(arrays, names, dance_energy, "dance_energy", 1)
        add_feature(arrays, names, duration_minutes, "duration_minutes", 20)
        add_feature(arrays, names, is_long_track, "is_long_track", 1)

        numeric_features = [
            ("danceability", "danceability", 1),
            ("energy", "energy", 1),
            ("loudness", "loudness", 30),
            ("speechiness", "speechiness", 1),
            ("acousticness", "acousticness", 1),
            ("instrumentalness", "instrumentalness", 1),
            ("liveness", "liveness", 1),
            ("valence", "valence", 1),
            ("tempo", "tempo", 300),
            ("duration_ms", "duration_ms", 600000),
            ("key", "key", 12),
            ("mode", "mode", 1),
            ("time_signature", "time_signature", 8),
            ("track_number", "track_number", 100),
            ("disc_number", "disc_number", 20),
        ]

        already_duration = False

        for requested, name, maximum in numeric_features:
            column = resolve_column(dataframe, requested)
            if column is None:
                continue

            if name == "duration" and already_duration:
                continue

            values = number_series(dataframe, requested).values
            add_feature(arrays, names, values, name, maximum)

            if name in ("duration_ms", "duration"):
                already_duration = True
                add_feature(
                    arrays,
                    names,
                    np.log1p(np.maximum(values, 0)),
                    name + "_log",
                    25,
                )

        explicit_col = resolve_first_column(dataframe, ["explicit", "Explicit"])
        if explicit_col is not None:
            explicit = (
                dataframe[explicit_col]
                .astype(str)
                .str.lower()
                .isin(["true", "1", "yes"])
                .astype(np.float32)
                .values
            )
        else:
            explicit = np.zeros(len(dataframe), dtype=np.float32)

        add_feature(arrays, names, explicit, "explicit", 1)

        # --- Genre Flags ---
        genre_lower = genre.str.lower()
        is_pop = genre_lower.str.contains(r"\bpop\b", na=False).astype(np.float32).values
        is_hiphop = genre_lower.str.contains(r"\b(?:hip hop|hip-hop|rap|trap)\b", na=False).astype(np.float32).values
        is_rock = genre_lower.str.contains(r"\b(?:rock|metal|punk|indie)\b", na=False).astype(np.float32).values
        is_edm = genre_lower.str.contains(r"\b(?:edm|electronic|dance|house|techno)\b", na=False).astype(np.float32).values

        add_feature(arrays, names, is_pop, "is_pop", 1)
        add_feature(arrays, names, is_hiphop, "is_hiphop", 1)
        add_feature(arrays, names, is_rock, "is_rock", 1)
        add_feature(arrays, names, is_edm, "is_edm", 1)

        # --- Metadata Features ---
        followers = number_series(dataframe, "artist_metadata_followers").values
        artist_popularity = number_series(dataframe, "artist_metadata_popularity").values
        graph_degree = number_series(dataframe, "artist_graph_degree").values
        graph_second_degree = number_series(dataframe, "artist_graph_second_degree").values
        graph_exists = number_series(dataframe, "artist_graph_exists").values

        add_feature(arrays, names, followers, "artist_followers", 1e9)
        add_feature(arrays, names, np.log1p(np.maximum(followers, 0)), "artist_followers_log", 30)
        add_feature(arrays, names, artist_popularity, "artist_popularity", 100)
        add_feature(arrays, names, graph_degree, "artist_graph_degree", 5000)
        add_feature(arrays, names, np.log1p(np.maximum(graph_degree, 0)), "artist_graph_degree_log", 20)
        add_feature(arrays, names, graph_second_degree, "artist_graph_second_degree", 10000)
        add_feature(arrays, names, np.log1p(np.maximum(graph_second_degree, 0)), "artist_graph_second_degree_log", 25)
        add_feature(arrays, names, graph_exists, "artist_graph_exists", 1)

        # --- Release Date Features ---
        release_col = resolve_first_column(dataframe, ["release_date", "release_date_normalized", "release", "date"])
        if release_col is not None:
            dates = pd.to_datetime(dataframe[release_col], errors="coerce")
            year = dates.dt.year.fillna(2000).astype(np.float32).values
            month = dates.dt.month.fillna(1).astype(np.float32).values
            decade = ((year // 10) * 10).astype(np.float32)
            
            # Recency bias is a massive factor in streaming platform popularity metrics
            years_since_2020 = np.clip(2020.0 - year, -50, 50).astype(np.float32)

            add_feature(arrays, names, year, "release_year", 2030)
            add_feature(arrays, names, month, "release_month", 12)
            add_feature(arrays, names, decade, "release_decade", 2030)
            add_feature(arrays, names, years_since_2020, "years_since_2020", 50)



    # =========================================================================
    # BOOK
    # =========================================================================

    elif category == "book":

        title = text_series(
            dataframe,
            "title",
        )

        author = text_series(
            dataframe,
            "author",
        )

        description = text_series(
            dataframe,
            "description",
        )

        genres = text_series(
            dataframe,
            "genres",
        )

        characters = text_series(
            dataframe,
            "characters",
        )

        awards = text_series(
            dataframe,
            "awards",
        )

        publisher = text_series(
            dataframe,
            "publisher",
        )

        series = text_series(
            dataframe,
            "series",
        )

        pages = number_series(
            dataframe,
            "pages",
        ).values

        num_ratings = number_series(
            dataframe,
            "numRatings",
        ).values

        liked_percent = number_series(
            dataframe,
            "likedPercent",
        ).values

        bbe_score = number_series(
            dataframe,
            "bbeScore",
        ).values

        bbe_votes = number_series(
            dataframe,
            "bbeVotes",
        ).values

        features = [

            (
                title.str.len().values,
                "title_length",
                500,
            ),

            (
                word_count(title),
                "title_words",
                100,
            ),

            (
                author.str.len().values,
                "author_length",
                500,
            ),

            (
                word_count(author),
                "author_words",
                100,
            ),

            (
                description.str.len().values,
                "description_length",
                10000,
            ),

            (
                word_count(description),
                "description_words",
                5000,
            ),

            (
                separator_count(genres),
                "genre_count",
                30,
            ),

            (
                separator_count(characters),
                "character_count",
                100,
            ),

            (
                separator_count(awards),
                "award_count",
                100,
            ),

            (
                publisher.str.len().values,
                "publisher_length",
                500,
            ),

            (
                series.str.len().values,
                "series_length",
                500,
            ),

            (
                pages,
                "pages",
                3000,
            ),

            (
                num_ratings,
                "num_ratings",
                1e8,
            ),

            (
                np.log1p(
                    np.maximum(
                        num_ratings,
                        0,
                    )
                ),
                "num_ratings_log",
                25,
            ),

            (
                liked_percent,
                "liked_percent",
                100,
            ),

            (
                bbe_score,
                "bbe_score",
                1e8,
            ),

            (
                np.log1p(
                    np.maximum(
                        bbe_score,
                        0,
                    )
                ),
                "bbe_score_log",
                25,
            ),

            (
                bbe_votes,
                "bbe_votes",
                1e8,
            ),

            (
                np.log1p(
                    np.maximum(
                        bbe_votes,
                        0,
                    )
                ),
                "bbe_votes_log",
                25,
            ),
        ]

        for value, name, maximum in features:

            add_feature(
                arrays,
                names,
                value,
                name,
                maximum,
            )

    # =========================================================================
    # APP
    # =========================================================================

    elif category == "app":

        reviews = number_series(
            dataframe,
            "Reviews",
        ).values

        installs = number_series(
            dataframe,
            "Installs",
        ).values

        price = number_series(
            dataframe,
            "Price",
        ).values

        size = number_series(
            dataframe,
            "Size",
        ).values

        app = text_series(
            dataframe,
            "App",
        )

        category_text = text_series(
            dataframe,
            "Category",
        )

        genres = text_series(
            dataframe,
            "Genres",
        )

        app_type = (
            text_series(
                dataframe,
                "Type",
            )
            .str.lower()
        )

        paid = (
            app_type
            .eq("paid")
            .astype(np.float32)
            .values
        )

        features = [

            (
                reviews,
                "reviews",
                1e8,
            ),

            (
                np.log1p(
                    np.maximum(
                        reviews,
                        0,
                    )
                ),
                "reviews_log",
                25,
            ),

            (
                installs,
                "installs",
                1e10,
            ),

            (
                np.log1p(
                    np.maximum(
                        installs,
                        0,
                    )
                ),
                "installs_log",
                30,
            ),

            (
                safe_ratio(
                    reviews,
                    installs,
                ),
                "review_install_ratio",
                1,
            ),

            (
                price,
                "price",
                500,
            ),

            (
                np.log1p(
                    np.maximum(
                        price,
                        0,
                    )
                ),
                "price_log",
                10,
            ),

            (
                paid,
                "paid_flag",
                1,
            ),

            (
                1.0 - paid,
                "free_flag",
                1,
            ),

            (
                size,
                "size",
                10000,
            ),

            (
                app.str.len().values,
                "app_name_length",
                500,
            ),

            (
                word_count(app),
                "app_name_words",
                100,
            ),

            (
                category_text.str.len().values,
                "category_length",
                100,
            ),

            (
                separator_count(genres),
                "genre_count",
                20,
            ),
        ]

        for value, name, maximum in features:

            add_feature(
                arrays,
                names,
                value,
                name,
                maximum,
            )

    # =========================================================================
    # GAME
    # =========================================================================

    elif category == "game":

        name = text_series(
            dataframe,
            "Name",
        )

        developer = text_series(
            dataframe,
            "Developer",
        )

        genre = text_series(
            dataframe,
            "Genre",
        )

        values = [

            (
                name.str.len().values,
                "name_length",
                500,
            ),

            (
                developer.str.len().values,
                "developer_length",
                500,
            ),

            (
                word_count(name),
                "name_words",
                100,
            ),

            (
                word_count(developer),
                "developer_words",
                100,
            ),

            (
                separator_count(genre),
                "genre_count",
                30,
            ),
        ]

        for value, name_value, maximum in values:

            add_feature(
                arrays,
                names,
                value,
                name_value,
                maximum,
            )

    # =========================================================================
    # MOVIE
    # =========================================================================

    elif category == "movie":

        values = [
            ("vote_count", "vote_count"),
            ("revenue", "revenue"),
            ("budget", "budget"),
            ("runtime", "runtime"),
            ("popularity", "popularity"),
        ]

        for requested, name_value in values:

            value = (
                number_series(
                    dataframe,
                    requested,
                )
                .values
            )

            add_feature(
                arrays,
                names,
                value,
                name_value,
            )

            add_feature(
                arrays,
                names,
                np.log1p(
                    np.maximum(
                        value,
                        0,
                    )
                ),
                name_value + "_log",
            )

    # =========================================================================
    # FINAL
    # =========================================================================

    if not arrays:

        return (
            np.zeros(
                (
                    len(dataframe),
                    1,
                ),
                dtype=np.float32,
            ),
            ["constant"],
        )

    matrix = (
        np.column_stack(
            arrays
        )
        .astype(np.float32)
    )

    matrix = np.nan_to_num(
        matrix,
        nan=0,
        posinf=0,
        neginf=0,
    )

    return (
        matrix,
        names,
    )


# =============================================================================
# MANUAL TF-IDF
# =============================================================================

def manual_sparse_tfidf(
    text_values,
    vectorizer,
):

    vocabulary = dict(
        vectorizer.vocabulary_
    )

    idf = np.asarray(
        vectorizer.idf_,
        dtype=np.float32,
    )

    analyzer = (
        vectorizer.build_analyzer()
    )

    rows = []
    cols = []
    data = []

    for row_index, value in enumerate(
        text_values.astype(str)
    ):

        counts = {}

        for token in analyzer(value):

            column_index = vocabulary.get(
                token
            )

            if column_index is None:
                continue

            counts[column_index] = (
                counts.get(
                    column_index,
                    0,
                )
                +
                1
            )

        for column_index, count in counts.items():

            if vectorizer.sublinear_tf:

                tf = (
                    1.0
                    +
                    math.log(
                        float(count)
                    )
                )

            else:

                tf = float(count)

            rows.append(
                row_index
            )

            cols.append(
                column_index
            )

            data.append(
                tf
                *
                float(
                    idf[
                        column_index
                    ]
                )
            )

    if not data:

        matrix = csr_matrix(
            (
                len(text_values),
                len(vocabulary),
            ),
            dtype=np.float32,
        )

    else:

        matrix = csr_matrix(
            (
                np.asarray(
                    data,
                    dtype=np.float32,
                ),
                (
                    np.asarray(
                        rows,
                        dtype=np.int32,
                    ),
                    np.asarray(
                        cols,
                        dtype=np.int32,
                    ),
                ),
            ),
            shape=(
                len(text_values),
                len(vocabulary),
            ),
            dtype=np.float32,
        )

    norms = np.sqrt(
        matrix.multiply(matrix)
        .sum(axis=1)
    ).A1

    norms[norms == 0] = 1

    matrix = matrix.multiply(
        (
            1.0 / norms
        )[:, None]
    ).tocsr()

    return matrix


# =============================================================================
# ROBUST SCALE
# =============================================================================

def fit_robust_scaler(values):

    values = np.asarray(
        values,
        dtype=np.float32,
    )

    if values.ndim != 2:

        values = values.reshape(
            len(values),
            -1,
        )

    center = np.median(
        values,
        axis=0,
    )

    q1 = np.percentile(
        values,
        25,
        axis=0,
    )

    q3 = np.percentile(
        values,
        75,
        axis=0,
    )

    scale = q3 - q1

    scale[
        ~np.isfinite(scale)
    ] = 1.0

    scale[
        scale == 0
    ] = 1.0

    center[
        ~np.isfinite(center)
    ] = 0.0

    return (
        center.astype(np.float32),
        scale.astype(np.float32),
    )


def apply_robust_scaler(
    values,
    center,
    scale,
):

    values = np.asarray(
        values,
        dtype=np.float32,
    )

    result = (
        values
        -
        center
    )

    result = (
        result
        /
        np.where(
            scale == 0,
            1.0,
            scale,
        )
    )

    return np.nan_to_num(
        result,
        nan=0,
        posinf=0,
        neginf=0,
    ).astype(np.float32)


# =============================================================================
# ENCODER HELPERS
# =============================================================================

def build_vectorizer_from_artifact(
    artifact
):

    class VectorizerHolder:
        pass

    holder = VectorizerHolder()

    holder.vocabulary_ = dict(
        artifact["vocabulary"]
    )

    holder.idf_ = np.asarray(
        artifact["idf"],
        dtype=np.float32,
    )

    holder.sublinear_tf = bool(
        artifact["sublinear_tf"]
    )

    # -------------------------------------------------------------------------
    # FIX:
    #
    # Do not depend on serializing a potentially complex analyzer callable.
    # Reconstruct the analyzer from its stored vectorizer settings.
    # -------------------------------------------------------------------------

    holder.ngram_range = tuple(
        artifact.get(
            "ngram_range",
            (1, 2),
        )
    )

    holder.lowercase = bool(
        artifact.get(
            "lowercase",
            True,
        )
    )

    holder.strip_accents = artifact.get(
        "strip_accents",
        "unicode",
    )

    holder.token_pattern = artifact.get(
        "token_pattern",
        r"(?u)\b\w\w+\b",
    )

    holder.analyzer_mode = artifact.get(
        "analyzer_mode",
        "word",
    )

    holder.stop_words = None

    holder.max_features = None
    holder.min_df = artifact.get(
        "min_df",
        1,
    )

    holder.max_df = artifact.get(
        "max_df",
        1.0,
    )

    holder.build_analyzer = (
        lambda:
        TfidfVectorizer(
            vocabulary=holder.vocabulary_,
            lowercase=holder.lowercase,
            strip_accents=holder.strip_accents,
            token_pattern=holder.token_pattern,
            ngram_range=holder.ngram_range,
            analyzer=holder.analyzer_mode,
            sublinear_tf=holder.sublinear_tf,
        ).build_analyzer()
    )

    return holder


# =============================================================================
# ENCODER
# =============================================================================

def fit_encoder(
    dataframe,
    category,
):

    text = build_text(
        dataframe,
        category,
    )

    max_features = (
        SONG_TFIDF_MAX_FEATURES
        if category == "song"
        else TFIDF_MAX_FEATURES
    )

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.995,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
    )

    try:

        vectorizer.fit(text)

    except ValueError:

        vectorizer = TfidfVectorizer(
            max_features=min(
                2000,
                max(
                    1,
                    len(text),
                ),
            ),
            min_df=1,
            token_pattern=r"(?u)\b\w+\b",
            sublinear_tf=True,
            ngram_range=(1, 2),
            strip_accents="unicode",
            lowercase=True,
        )

        vectorizer.fit(text)

    sparse = manual_sparse_tfidf(
        text,
        vectorizer,
    )

    if sparse.shape[1] == 0:

        sparse = csr_matrix(
            (
                len(text),
                1,
            ),
            dtype=np.float32,
        )

    component_limit = min(
        SVD_COMPONENTS,
        max(
            1,
            sparse.shape[0] - 1,
        ),
        max(
            1,
            sparse.shape[1] - 1,
        ),
    )

    components = None

    if (
        component_limit >= 2
        and
        sparse.shape[0] >= 3
        and
        sparse.shape[1] >= 3
    ):

        svd = TruncatedSVD(
            n_components=component_limit,
            random_state=SEED,
        )

        svd.fit(sparse)

        components = np.asarray(
            svd.components_,
            dtype=np.float32,
        )

    if components is not None:

        semantic = np.asarray(
            sparse.dot(
                components.T
            ),
            dtype=np.float32,
        )

    else:

        semantic = (
            sparse
            .toarray()
            .astype(np.float32)
        )

    structured, schema = make_features(
        dataframe,
        category,
    )

    center, scale = fit_robust_scaler(
        structured
    )

    structured_scaled = (
        apply_robust_scaler(
            structured,
            center,
            scale,
        )
    )

    X = np.hstack(
        [
            semantic,
            structured_scaled,
        ]
    ).astype(np.float32)

    artifact = {

        "vocabulary":
            dict(
                vectorizer.vocabulary_
            ),

        "idf":
            np.asarray(
                vectorizer.idf_,
                dtype=np.float32,
            ),

        "sublinear_tf":
            bool(
                vectorizer.sublinear_tf
            ),

        "ngram_range":
            tuple(
                vectorizer.ngram_range
            ),

        "lowercase":
            bool(
                vectorizer.lowercase
            ),

        "strip_accents":
            vectorizer.strip_accents,

        "token_pattern":
            vectorizer.token_pattern,

        "analyzer_mode":
            vectorizer.analyzer,

        "components":
            components,

        "schema":
            list(schema),

        "center":
            center,

        "scale":
            scale,
    }

    return X, artifact


# =============================================================================
# ENCODE DATA
# =============================================================================

def encode_data(
    dataframe,
    category,
    artifact,
):

    holder = build_vectorizer_from_artifact(
        artifact
    )

    text = build_text(
        dataframe,
        category,
    )

    sparse = manual_sparse_tfidf(
        text,
        holder,
    )

    components = artifact[
        "components"
    ]

    if components is not None:

        semantic = np.asarray(
            sparse.dot(
                components.T
            ),
            dtype=np.float32,
        )

    else:

        semantic = (
            sparse
            .toarray()
            .astype(np.float32)
        )

    structured, names = make_features(
        dataframe,
        category,
    )

    source = {
        name:
            structured[:, index]
        for index, name
        in enumerate(names)
    }

    aligned = np.zeros(
        (
            len(dataframe),
            len(
                artifact["schema"]
            ),
        ),
        dtype=np.float32,
    )

    for index, name in enumerate(
        artifact["schema"]
    ):

        if name in source:

            aligned[:, index] = (
                source[name]
            )

    structured_scaled = (
        apply_robust_scaler(
            aligned,
            artifact["center"],
            artifact["scale"],
        )
    )

    X = np.hstack(
        [
            semantic,
            structured_scaled,
        ]
    ).astype(np.float32)

    return np.nan_to_num(
        X,
        nan=0,
        posinf=0,
        neginf=0,
    )


# =============================================================================
# METRICS
# =============================================================================

def ndcg10(y, prediction):

    if len(y) == 0:
        return 0.0

    y = np.asarray(
        y,
        dtype=np.float64,
    )

    prediction = np.asarray(
        prediction,
        dtype=np.float64,
    )

    y = np.nan_to_num(
        y,
        nan=0,
        posinf=0,
        neginf=0,
    )

    prediction = np.nan_to_num(
        prediction,
        nan=0,
        posinf=0,
        neginf=0,
    )

    k = min(
        10,
        len(y),
    )

    predicted = np.argsort(
        -prediction
    )[:k]

    ideal = np.argsort(
        -y
    )[:k]

    gains = np.maximum(
        y - np.min(y),
        0,
    )

    discounts = np.log2(
        np.arange(k) + 2
    )

    dcg = np.sum(
        gains[predicted]
        /
        discounts
    )

    idcg = np.sum(
        gains[ideal]
        /
        discounts
    )

    if idcg <= 1e-12:
        return 0.0

    return float(
        dcg / idcg
    )


def calculate_metrics(
    y,
    prediction,
):

    y = np.asarray(
        y,
        dtype=np.float64,
    )

    prediction = np.asarray(
        prediction,
        dtype=np.float64,
    )

    y = np.nan_to_num(
        y,
        nan=0,
        posinf=0,
        neginf=0,
    )

    prediction = np.nan_to_num(
        prediction,
        nan=0,
        posinf=0,
        neginf=0,
    )

    if np.unique(y).size > 1:

        r2 = float(
            r2_score(
                y,
                prediction,
            )
        )

    else:

        r2 = 0.0

    return {

        "mae":
            float(
                mean_absolute_error(
                    y,
                    prediction,
                )
            ),

        "rmse":
            float(
                math.sqrt(
                    mean_squared_error(
                        y,
                        prediction,
                    )
                )
            ),

        "r2":
            r2,

        "ndcg10":
            ndcg10(
                y,
                prediction,
            ),
    }


# =============================================================================
# MODELS
# =============================================================================

def make_tree(category):

    settings = TREE_SETTINGS[
        category
    ]

    return DecisionTreeRegressor(
        criterion="squared_error",

        max_depth=settings[
            "max_depth"
        ],

        min_samples_leaf=settings[
            "min_samples_leaf"
        ],

        min_samples_split=settings[
            "min_samples_split"
        ],

        max_features=0.90,

        random_state=SEED,
    )


def make_mlp(category):

    settings = MLP_SETTINGS[
        category
    ]

    return MLPRegressor(
        hidden_layer_sizes=settings[
            "hidden"
        ],

        activation="relu",

        solver="adam",

        alpha=settings[
            "alpha"
        ],

        batch_size=128,

        learning_rate_init=0.0008,

        max_iter=settings[
            "max_iter"
        ],

        early_stopping=True,

        validation_fraction=0.12,

        n_iter_no_change=12,

        tol=1e-4,

        random_state=SEED,

        verbose=False,
    )


def safe_fit_mlp(
    model,
    X,
    y,
):

    # Very small datasets can make early stopping unstable.
    if len(X) < 30:

        model.set_params(
            early_stopping=False,
        )

    try:

        model.fit(
            X,
            y,
        )

    except ValueError:

        # Last-resort safe retry for tiny / unusual datasets.
        model.set_params(
            early_stopping=False,
        )

        model.fit(
            X,
            y,
        )

    return model


# =============================================================================
# META
# =============================================================================

def build_meta_matrix(
    tree,
    mlp,
):

    tree = np.asarray(
        tree,
        dtype=np.float32,
    )

    mlp = np.asarray(
        mlp,
        dtype=np.float32,
    )

    return np.column_stack(
        [
            tree,
            mlp,
            np.abs(
                tree - mlp
            ),
            (
                tree + mlp
            ) / 2.0,
        ]
    ).astype(np.float32)


def fit_meta(
    tree_oof,
    mlp_oof,
    y,
):

    meta_X = build_meta_matrix(
        tree_oof,
        mlp_oof,
    )

    model = Ridge(
        alpha=4.0
    )

    model.fit(
        meta_X,
        y,
    )

    return model


# =============================================================================
# OOF
# =============================================================================

def generate_oof(
    category,
    dataframe,
    y,
):

    n = len(dataframe)

    if n < 3:

        mean_value = (
            float(
                np.mean(y)
            )
            if n
            else 0.0
        )

        return (
            np.full(
                n,
                mean_value,
                dtype=np.float32,
            ),

            np.full(
                n,
                mean_value,
                dtype=np.float32,
            ),
        )

    folds_count = min(
        OOF_FOLDS,
        n,
    )

    tree_oof = np.zeros(
        n,
        dtype=np.float32,
    )

    mlp_oof = np.zeros(
        n,
        dtype=np.float32,
    )

    rng = np.random.default_rng(
        SEED
    )

    indices = np.arange(n)

    rng.shuffle(indices)

    folds = np.array_split(
        indices,
        folds_count,
    )

    for fold_id in range(
        folds_count
    ):

        print(
            "      OOF fold "
            f"{fold_id + 1}/{folds_count}"
        )

        valid_idx = folds[
            fold_id
        ]

        train_parts = [
            folds[index]
            for index in range(
                folds_count
            )
            if index != fold_id
        ]

        train_idx = np.concatenate(
            train_parts
        )

        train_df = (
            dataframe
            .iloc[train_idx]
            .reset_index(drop=True)
        )

        valid_df = (
            dataframe
            .iloc[valid_idx]
            .reset_index(drop=True)
        )

        y_train = y[
            train_idx
        ]

        X_train, encoder = fit_encoder(
            train_df,
            category,
        )

        X_valid = encode_data(
            valid_df,
            category,
            encoder,
        )

        # ---------------------------------------------------------------------
        # TREE
        # ---------------------------------------------------------------------

        tree = make_tree(
            category
        )

        tree.fit(
            X_train,
            y_train,
        )

        tree_oof[
            valid_idx
        ] = np.clip(
            tree.predict(
                X_valid
            ),
            0,
            1,
        )

        # ---------------------------------------------------------------------
        # MLP
        # ---------------------------------------------------------------------

        if USE_MLP:

            mlp_center, mlp_scale = (
                fit_robust_scaler(
                    X_train
                )
            )

            X_train_scaled = (
                apply_robust_scaler(
                    X_train,
                    mlp_center,
                    mlp_scale,
                )
            )

            X_valid_scaled = (
                apply_robust_scaler(
                    X_valid,
                    mlp_center,
                    mlp_scale,
                )
            )

            mlp = make_mlp(
                category
            )

            safe_fit_mlp(
                mlp,
                X_train_scaled,
                y_train,
            )

            mlp_oof[
                valid_idx
            ] = np.clip(
                mlp.predict(
                    X_valid_scaled
                ),
                0,
                1,
            )

        else:

            mlp_oof[
                valid_idx
            ] = np.mean(
                y_train
            )

        del (
            train_df,
            valid_df,
            X_train,
            X_valid,
            tree,
            encoder,
        )

        gc.collect()

    return (
        tree_oof,
        mlp_oof,
    )


# =============================================================================
# GATE
# =============================================================================

def gate_score(report):

    return (
        0.65 * report["mae"]
        +
        0.25 * report["rmse"]
        +
        0.10 *
        (
            1.0
            -
            report["ndcg10"]
        )
    )


def independent_gate(
    category,
    train_df,
    y,
):

    print()
    print(
        "Creating independent gate set..."
    )

    if len(train_df) < 10:

        print(
            "Dataset too small for normal gate; "
            "using direct model comparison."
        )

    base_idx, gate_idx = split_indices(
        y,
        GATE_SIZE,
        SEED + 100,
    )

    base_df = (
        train_df
        .iloc[base_idx]
        .reset_index(drop=True)
    )

    gate_df = (
        train_df
        .iloc[gate_idx]
        .reset_index(drop=True)
    )

    y_base = y[
        base_idx
    ]

    y_gate = y[
        gate_idx
    ]

    print(
        "Gate base:",
        len(base_df),
        "| independent gate:",
        len(gate_df),
    )

    X_base, encoder = fit_encoder(
        base_df,
        category,
    )

    X_gate = encode_data(
        gate_df,
        category,
        encoder,
    )

    # =========================================================================
    # TREE
    # =========================================================================

    tree = make_tree(
        category
    )

    tree.fit(
        X_base,
        y_base,
    )

    gate_tree = np.clip(
        tree.predict(
            X_gate
        ),
        0,
        1,
    )

    # =========================================================================
    # MLP
    # =========================================================================

    if USE_MLP:

        center, scale = fit_robust_scaler(
            X_base
        )

        X_base_scaled = (
            apply_robust_scaler(
                X_base,
                center,
                scale,
            )
        )

        X_gate_scaled = (
            apply_robust_scaler(
                X_gate,
                center,
                scale,
            )
        )

        mlp = make_mlp(
            category
        )

        safe_fit_mlp(
            mlp,
            X_base_scaled,
            y_base,
        )

        gate_mlp = np.clip(
            mlp.predict(
                X_gate_scaled
            ),
            0,
            1,
        )

    else:

        center = np.zeros(
            X_base.shape[1],
            dtype=np.float32,
        )

        scale = np.ones(
            X_base.shape[1],
            dtype=np.float32,
        )

        mlp = None

        gate_mlp = np.full(
            len(gate_df),
            np.mean(y_base),
            dtype=np.float32,
        )

    # =========================================================================
    # META
    #
    # IMPORTANT:
    # Train meta on OOF predictions from base data instead of using the same
    # model's in-sample predictions.
    # =========================================================================

    if len(base_df) >= 6:

        base_tree_oof, base_mlp_oof = (
            generate_oof(
                category,
                base_df,
                y_base,
            )
        )

        meta = fit_meta(
            base_tree_oof,
            base_mlp_oof,
            y_base,
        )

    else:

        meta = fit_meta(
            np.full(
                len(y_base),
                np.mean(y_base),
                dtype=np.float32,
            ),
            np.full(
                len(y_base),
                np.mean(y_base),
                dtype=np.float32,
            ),
            y_base,
        )

    gate_meta = np.clip(
        meta.predict(
            build_meta_matrix(
                gate_tree,
                gate_mlp,
            )
        ),
        0,
        1,
    )

    # =========================================================================
    # REPORTS
    # =========================================================================

    reports = {

        "tree":
            calculate_metrics(
                y_gate,
                gate_tree,
            ),

        "meta":
            calculate_metrics(
                y_gate,
                gate_meta,
            ),
    }

    if USE_MLP:

        reports["mlp"] = calculate_metrics(
            y_gate,
            gate_mlp,
        )

    scores = {
        name:
            gate_score(report)
        for name, report
        in reports.items()
    }

    ranked = sorted(
        reports.keys(),
        key=lambda name: (
            reports[name]["mae"],
            reports[name]["rmse"],
            -reports[name]["ndcg10"],
        ),
    )

    selected = ranked[0]

    # =========================================================================
    # MLP DOMINANCE PROTECTION
    # =========================================================================

    if (
        selected == "mlp"
        and
        len(ranked) > 1
    ):

        non_mlp = [
            name
            for name in ranked
            if name != "mlp"
        ]

        best_non_mlp = min(
            non_mlp,
            key=lambda name:
                reports[name]["mae"],
        )

        if (
            reports["mlp"]["mae"]
            >
            reports[best_non_mlp]["mae"]
            *
            MLP_DOMINANCE_FACTOR
        ):

            selected = best_non_mlp

    print()
    print(
        "V26 STABLE INDEPENDENT MODEL GATE"
    )

    for name in (
        "tree",
        "mlp",
        "meta",
    ):

        if name not in reports:
            continue

        report = reports[name]

        print(
            "{} | "
            "MAE={:.5f} | "
            "RMSE={:.5f} | "
            "R2={:.5f} | "
            "NDCG={:.5f} | "
            "SCORE={:.5f}".format(
                name.upper(),
                report["mae"],
                report["rmse"],
                report["r2"],
                report["ndcg10"],
                scores[name],
            )
        )

    print(
        "RANK:",
        " > ".join(ranked)
    )

    print(
        "SELECTED:",
        selected,
    )

    del (
        base_df,
        gate_df,
        X_base,
        X_gate,
        tree,
        meta,
        encoder,
    )

    gc.collect()

    return {
        "selected": selected,
        "reports": reports,
        "scores": scores,
        "ranked": ranked,
    }


# =============================================================================
# DEMO
# =============================================================================

def make_demo(category):

    if category == "song":

        return pd.DataFrame(
            [
                {
                    "artist": "ABBA",
                    "song": "Dancing Queen",
                    "name": "Dancing Queen",
                    "album": "Arrival",
                    "genre": "Pop Disco",
                    "artist_metadata_genres": "pop disco",
                    "artist_metadata_followers": 1000000,
                    "artist_metadata_popularity": 80,
                    "artist_graph_degree": 10,
                    "artist_graph_second_degree": 20,
                    "artist_graph_exists": 1,
                    "danceability": 0.75,
                    "energy": 0.70,
                    "valence": 0.80,
                    "tempo": 100,
                    "duration_ms": 230000,
                },

                {
                    "artist": "Queen",
                    "song": "Bohemian Rhapsody",
                    "name": "Bohemian Rhapsody",
                    "album": "A Night at the Opera",
                    "genre": "Rock",
                    "artist_metadata_genres": "rock classic rock",
                    "artist_metadata_followers": 1500000,
                    "artist_metadata_popularity": 85,
                    "artist_graph_degree": 15,
                    "artist_graph_second_degree": 30,
                    "artist_graph_exists": 1,
                    "danceability": 0.40,
                    "energy": 0.55,
                    "valence": 0.30,
                    "tempo": 72,
                    "duration_ms": 355000,
                },
            ]
        )

    if category == "book":

        return pd.DataFrame(
            [
                {
                    "title":
                        "The Hunger Games",
                    "series":
                        "The Hunger Games #1",
                    "author":
                        "Suzanne Collins",
                    "rating":
                        4.33,
                    "description":
                        "A young heroine enters a dangerous televised competition in a dystopian society.",
                    "language":
                        "English",
                    "genres":
                        "Young Adult Fiction Dystopia Fantasy Science Fiction",
                    "characters":
                        "Katniss Everdeen Peeta Mellark",
                    "publisher":
                        "Scholastic Press",
                    "pages":
                        374,
                    "numRatings":
                        6376780,
                    "likedPercent":
                        96,
                    "bbeScore":
                        2993816,
                    "bbeVotes":
                        30516,
                    "awards":
                        "",
                },
            ]
        )

    if category == "app":

        return pd.DataFrame(
            [
                {
                    "App":
                        "Study Planner",
                    "Category":
                        "Education",
                    "Type":
                        "Free",
                    "Reviews":
                        12500,
                    "Installs":
                        "500,000+",
                    "Price":
                        "Free",
                    "Genres":
                        "Education",
                },
            ]
        )

    if category == "game":

        return pd.DataFrame(
            [
                {
                    "Name":
                        "Neon Kingdom",
                    "Developer":
                        "Future Games",
                    "Genre":
                        "Action Adventure",
                    "Date Released":
                        "2022-08-15",
                },
            ]
        )

    return pd.DataFrame(
        [
            {
                "title":
                    "Beyond the Winter Sea",
                "status":
                    "Released",
                "original_language":
                    "en",
                "original_title":
                    "Beyond the Winter Sea",
                "overview":
                    "Two researchers search for a lost expedition on a frozen coast.",
                "tagline":
                    "The journey changes everything.",
                "genres":
                    "Adventure Drama",
                "production_companies":
                    "Independent",
                "production_countries":
                    "Canada",
                "spoken_languages":
                    "English",
                "keywords":
                    "winter expedition ocean",
                "vote_count":
                    900,
                "revenue":
                    55000000,
                "runtime":
                    126,
                "budget":
                    21000000,
                "popularity":
                    62,
                "vote_average":
                    7.0,
            },
        ]
    )


# =============================================================================
# MOVIE STREAM LOADER
# =============================================================================

def load_movie():

    path = DATASETS["movie"]

    if not path.exists():

        raise FileNotFoundError(
            str(path)
        )

    print(
        "Streaming Movie dataset..."
    )

    reservoir = []

    rng = np.random.default_rng(
        SEED
    )

    seen = 0
    header = None

    for chunk in pd.read_csv(
        path,
        chunksize=50000,
        low_memory=False,
    ):

        if header is None:

            header = list(
                chunk.columns
            )

        for row in chunk.itertuples(
            index=False,
            name=None,
        ):

            seen += 1

            if len(reservoir) < MAX_ROWS:

                reservoir.append(row)

            else:

                position = int(
                    rng.integers(
                        0,
                        seen,
                    )
                )

                if position < MAX_ROWS:

                    reservoir[position] = row

        if seen % 250000 == 0:

            print(
                "rows scanned:",
                seen,
            )

        del chunk

        gc.collect()

    dataframe = pd.DataFrame(
        reservoir,
        columns=header,
    )

    return (
        dataframe,
        path,
    )


# =============================================================================
# LOAD DATASET
# =============================================================================

def load_dataset(category):

    if category == "song":

        return load_song_dataset()

    if category == "movie":

        return load_movie()

    path = DATASETS[category]

    if not path.exists():

        raise FileNotFoundError(
            str(path)
        )

    dataframe = pd.read_csv(
        path,
        low_memory=False,
    )

    return (
        dataframe,
        path,
    )


# =============================================================================
# TRAIN DOMAIN
# =============================================================================

def train_domain(
    category,
    dataframe,
    source,
):

    start = time.time()

    print()
    print(
        "=" * 110
    )

    print(
        "DECISION SYSTEM v26 FIXED:",
        category.upper(),
    )

    print(
        "=" * 110
    )

    before = len(dataframe)

    dataframe = (
        dataframe
        .drop_duplicates()
        .reset_index(drop=True)
    )

    print(
        "Duplicates removed:",
        before - len(dataframe),
    )

    if len(dataframe) > MAX_ROWS:

        dataframe = (
            dataframe
            .sample(
                n=MAX_ROWS,
                random_state=SEED,
            )
            .reset_index(drop=True)
        )

    print(
        "Rows used:",
        len(dataframe),
    )

    if len(dataframe) < 3:

        raise ValueError(
            "Not enough valid rows for training."
        )

    target, valid, target_name = (
        build_target(
            dataframe,
            category,
        )
    )

    invalid = int(
        np.sum(~valid)
    )

    if invalid:

        print(
            "Invalid target rows removed:",
            invalid,
        )

        dataframe = (
            dataframe
            .loc[valid]
            .reset_index(drop=True)
        )

        target = target[valid]

    if len(dataframe) < 3:

        raise ValueError(
            "Not enough valid target rows."
        )

    target_type = (
        "SYNTHETIC"
        if category == "game"
        else "REAL"
    )

    print(
        "Target:",
        target_name,
    )

    print(
        "Target type:",
        target_type,
    )

    print(
        "Feature mode:",
        "V26 MULTI-SOURCE FIXED",
    )

    train_idx, test_idx = split_indices(
        target,
        OUTER_TEST_SIZE,
        SEED,
    )

    train_df = (
        dataframe
        .iloc[train_idx]
        .reset_index(drop=True)
    )

    test_df = (
        dataframe
        .iloc[test_idx]
        .reset_index(drop=True)
    )

    y_train_raw = target[
        train_idx
    ]

    y_test_raw = target[
        test_idx
    ]

    scaler = (
        TargetScaler()
        .fit(y_train_raw)
    )

    y_train = scaler.normalize(
        y_train_raw
    )

    y_test = scaler.normalize(
        y_test_raw
    )

    print(
        "Train:",
        len(train_df),
        "| Test:",
        len(test_df),
    )

    # =========================================================================
    # GATE
    # =========================================================================

    gate = independent_gate(
        category,
        train_df,
        y_train,
    )

    selected = gate["selected"]

    # =========================================================================
    # FINAL ENCODER
    # =========================================================================

    print()
    print(
        "Training final feature pipeline..."
    )

    X_train, encoder = fit_encoder(
        train_df,
        category,
    )

    print(
        "Final feature count:",
        X_train.shape[1],
    )

    # =========================================================================
    # FINAL OOF
    # =========================================================================

    print()
    print(
        "Creating final OOF predictions..."
    )

    tree_oof, mlp_oof = generate_oof(
        category,
        train_df,
        y_train,
    )

    meta = fit_meta(
        tree_oof,
        mlp_oof,
        y_train,
    )

    # =========================================================================
    # FINAL TREE
    # =========================================================================

    print(
        "Training final Decision Tree..."
    )

    final_tree = make_tree(
        category
    )

    final_tree.fit(
        X_train,
        y_train,
    )

    # =========================================================================
    # FINAL MLP
    # =========================================================================

    print(
        "Training final Small MLP..."
    )

    if USE_MLP:

        mlp_center, mlp_scale = (
            fit_robust_scaler(
                X_train
            )
        )

        X_train_scaled = (
            apply_robust_scaler(
                X_train,
                mlp_center,
                mlp_scale,
            )
        )

        final_mlp = make_mlp(
            category
        )

        safe_fit_mlp(
            final_mlp,
            X_train_scaled,
            y_train,
        )

    else:

        mlp_center = np.zeros(
            X_train.shape[1],
            dtype=np.float32,
        )

        mlp_scale = np.ones(
            X_train.shape[1],
            dtype=np.float32,
        )

        final_mlp = None

    # =========================================================================
    # TEST
    # =========================================================================

    X_test = encode_data(
        test_df,
        category,
        encoder,
    )

    tree_test = np.clip(
        final_tree.predict(
            X_test
        ),
        0,
        1,
    )

    if USE_MLP:

        X_test_scaled = (
            apply_robust_scaler(
                X_test,
                mlp_center,
                mlp_scale,
            )
        )

        mlp_test = np.clip(
            final_mlp.predict(
                X_test_scaled
            ),
            0,
            1,
        )

    else:

        mlp_test = np.full(
            len(X_test),
            np.mean(y_train),
            dtype=np.float32,
        )

    meta_test = np.clip(
        meta.predict(
            build_meta_matrix(
                tree_test,
                mlp_test,
            )
        ),
        0,
        1,
    )

    # =========================================================================
    # REPORT
    # =========================================================================

    reports = {

        "tree":
            calculate_metrics(
                y_test,
                tree_test,
            ),

        "meta":
            calculate_metrics(
                y_test,
                meta_test,
            ),
    }

    if USE_MLP:

        reports["mlp"] = calculate_metrics(
            y_test,
            mlp_test,
        )

    if selected == "tree":

        final_prediction = tree_test

    elif selected == "meta":

        final_prediction = meta_test

    else:

        final_prediction = mlp_test

    final_report = calculate_metrics(
        y_test,
        final_prediction,
    )

    print()
    print(
        "=" * 100
    )

    print(
        "UNSEEN TEST"
    )

    for name in (
        "tree",
        "mlp",
        "meta",
    ):

        if name not in reports:
            continue

        print(
            name.upper() + ":",
            reports[name],
        )

    print(
        "SELECTED:",
        selected,
    )

    print(
        "FINAL:",
        final_report,
    )

    print(
        "=" * 100
    )

    # =========================================================================
    # TOP 30
    # =========================================================================

    order = (
        np.argsort(
            -final_prediction
        )[
            :
            min(
                TOP_K,
                len(test_df),
            )
        ]
    )

    top = (
        test_df
        .iloc[order]
        .copy()
    )

    top.insert(
        0,
        "rank",
        np.arange(
            len(top)
        ) + 1,
    )

    top["decision_score"] = (
        final_prediction[order]
    )

    top["predicted_target"] = (
        scaler.denormalize(
            final_prediction[order]
        )
    )

    top["tree_prediction"] = (
        scaler.denormalize(
            tree_test[order]
        )
    )

    top["meta_prediction"] = (
        scaler.denormalize(
            meta_test[order]
        )
    )

    if USE_MLP:

        top["mlp_prediction"] = (
            scaler.denormalize(
                mlp_test[order]
            )
        )

    top_path = (
        OUTPUT
        /
        f"{category}_top30.csv"
    )

    top.to_csv(
        top_path,
        index=False,
        encoding="utf-8-sig",
    )

    # =========================================================================
    # NEW ITEMS
    # =========================================================================

    demo = make_demo(
        category
    )

    demo_X = encode_data(
        demo,
        category,
        encoder,
    )

    demo_tree = np.clip(
        final_tree.predict(
            demo_X
        ),
        0,
        1,
    )

    if USE_MLP:

        demo_scaled = (
            apply_robust_scaler(
                demo_X,
                mlp_center,
                mlp_scale,
            )
        )

        demo_mlp = np.clip(
            final_mlp.predict(
                demo_scaled
            ),
            0,
            1,
        )

    else:

        demo_mlp = np.full(
            len(demo_X),
            np.mean(y_train),
            dtype=np.float32,
        )

    demo_meta = np.clip(
        meta.predict(
            build_meta_matrix(
                demo_tree,
                demo_mlp,
            )
        ),
        0,
        1,
    )

    if selected == "tree":

        demo_final = demo_tree

    elif selected == "meta":

        demo_final = demo_meta

    else:

        demo_final = demo_mlp

    demo_output = demo.copy()

    demo_output["decision_score"] = (
        demo_final
    )

    demo_output["predicted_target"] = (
        scaler.denormalize(
            demo_final
        )
    )

    demo_output = (
        demo_output
        .sort_values(
            "decision_score",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    demo_output.insert(
        0,
        "rank",
        np.arange(
            len(demo_output)
        ) + 1,
    )

    demo_path = (
        OUTPUT
        /
        f"{category}_new_item_predictions.csv"
    )

    demo_output.to_csv(
        demo_path,
        index=False,
        encoding="utf-8-sig",
    )

    # =========================================================================
    # SAVE MODEL
    # =========================================================================

    model_path = (
        OUTPUT
        /
        f"{category}_model.joblib"
    )

    artifact = {

        "version":
            "DecisionSystem-v26-FIXED",

        "category":
            category,

        "source":
            str(source),

        "target":
            target_name,

        "target_type":
            target_type,

        "encoder":
            encoder,

        "target_scaler":
            scaler,

        "mlp_center":
            mlp_center,

        "mlp_scale":
            mlp_scale,

        "tree":
            final_tree,

        "mlp":
            final_mlp,

        "meta":
            meta,

        "selected":
            selected,

        "gate":
            gate,

        "use_mlp":
            USE_MLP,

        "max_rows":
            MAX_ROWS,

        "cpu_threads":
            CPU_THREADS,
    }

    joblib.dump(
        artifact,
        model_path,
        compress=3,
    )

    # =========================================================================
    # RESULT JSON
    # =========================================================================

    elapsed = (
        time.time()
        -
        start
    )

    result = {

        "version":
            "DecisionSystem-v26-FIXED",

        "category":
            category,

        "source":
            str(source),

        "target":
            target_name,

        "target_type":
            target_type,

        "rows":
            int(len(dataframe)),

        "train_rows":
            int(len(train_df)),

        "test_rows":
            int(len(test_df)),

        "features":
            int(X_train.shape[1]),

        "selected":
            selected,

        "gate":
            gate,

        "test":
            reports,

        "final":
            final_report,

        "files":
            {
                "model":
                    str(model_path),

                "top30":
                    str(top_path),

                "new_items":
                    str(demo_path),
            },

        "elapsed_seconds":
            round(
                elapsed,
                2,
            ),
    }

    result_path = (
        OUTPUT
        /
        f"{category}_result.json"
    )

    with open(
        result_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2,
            default=float,
        )

    # =========================================================================
    # FINAL PRINT
    # =========================================================================

    print()
    print(
        "=" * 110
    )

    print(
        category.upper(),
        "FINAL RESULT"
    )

    print(
        "=" * 110
    )

    for name in (
        "tree",
        "mlp",
        "meta",
    ):

        if name not in reports:
            continue

        report = reports[name]

        print(
            "{} | "
            "MAE={:.5f} | "
            "RMSE={:.5f} | "
            "R2={:.5f} | "
            "NDCG={:.5f}".format(
                name.upper(),
                report["mae"],
                report["rmse"],
                report["r2"],
                report["ndcg10"],
            )
        )

    print()

    print(
        "FINAL | "
        "MAE={:.5f} | "
        "RMSE={:.5f} | "
        "R2={:.5f} | "
        "NDCG={:.5f}".format(
            final_report["mae"],
            final_report["rmse"],
            final_report["r2"],
            final_report["ndcg10"],
        )
    )

    print()

    print(
        "Selected:",
        selected,
    )

    print(
        "Features:",
        X_train.shape[1],
    )

    print(
        "Time:",
        f"{elapsed:.1f} sec",
    )

    print(
        "Output:",
        OUTPUT,
    )

    print(
        "=" * 110
    )

    return result


# =============================================================================
# MAIN
# =============================================================================

def main():

    print()
    print(
        "=" * 130
    )

    print(
        " DECISION SYSTEM v26 FIXED"
    )

    print(
        " SPOTIFY TRACKS + ARTISTS + ARTIST GRAPH"
    )

    print(
        " MANUAL TF-IDF + SVD + TREE + SMALL MLP + META"
    )

    print(
        " NO .transform()"
    )

    print(
        " MAX 5000 ROWS / DOMAIN"
    )

    print(
        " CPU THREAD LIMIT = 2"
    )

    print(
        " ARTIST JOIN FIXED"
    )

    print(
        " META OOF FIXED"
    )

    print(
        "=" * 130
    )

    print()

    print(
        "CPU threads:",
        CPU_THREADS,
    )

    print(
        "Maximum rows/domain:",
        MAX_ROWS,
    )

    print(
        "OOF folds:",
        OOF_FOLDS,
    )

    print(
        "MLP enabled:",
        USE_MLP,
    )

    results = []

    for category in (
        "song",
        "app",
        "game",
        "book",
        "movie",
    ):

        print()
        print(
            "#" * 130
        )

        print(
            "LOADING:",
            category.upper(),
        )

        print(
            "#" * 130
        )

        try:

            dataframe, source = (
                load_dataset(
                    category
                )
            )

            print(
                "Loaded:",
                len(dataframe),
                "rows |",
                len(dataframe.columns),
                "columns",
            )

            result = train_domain(
                category,
                dataframe,
                source,
            )

            results.append(
                result
            )

            del dataframe

            gc.collect()

        except Exception as error:

            print()
            print(
                category.upper(),
                "FAILED:"
            )

            print(
                repr(error)
            )

            import traceback

            traceback.print_exc()

            gc.collect()

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print()
    print(
        "=" * 160
    )

    print(
        " DECISION SYSTEM v26 FIXED — FINAL SUMMARY"
    )

    print(
        "=" * 160
    )

    print(
        "{:<10} "
        "{:<32} "
        "{:<10} "
        "{:>10} "
        "{:>10} "
        "{:>10} "
        "{:>12}".format(
            "DOMAIN",
            "TARGET",
            "MODEL",
            "TREE_R2",
            "MLP_R2",
            "META_R2",
            "FINAL_NDCG",
        )
    )

    print(
        "-" * 160
    )

    for result in results:

        reports = result["test"]

        final = result["final"]

        tree_r2 = reports[
            "tree"
        ]["r2"]

        meta_r2 = reports[
            "meta"
        ]["r2"]

        mlp_r2 = (
            reports["mlp"]["r2"]
            if "mlp" in reports
            else 0.0
        )

        print(
            "{:<10} "
            "{:<32} "
            "{:<10} "
            "{:>10.4f} "
            "{:>10.4f} "
            "{:>10.4f} "
            "{:>12.4f}".format(
                result["category"].upper(),
                result["target"],
                result["selected"],
                tree_r2,
                mlp_r2,
                meta_r2,
                final["ndcg10"],
            )
        )

    print(
        "=" * 160
    )

    print()

    print(
        "FINISHED:",
        len(results),
        "/5",
    )

    print(
        "Output:",
        OUTPUT,
    )


# =============================================================================
# ENTRY
# =============================================================================

if __name__ == "__main__":
    main()