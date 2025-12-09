from __future__ import annotations

import pandas as pd
import numpy as np
import math
from typing import Dict, List, Tuple, Union
from sklearn.datasets import fetch_california_housing, load_diabetes, load_iris, load_breast_cancer

from faex.mathing.RandomGenerator import RandomGenerator

def bikes(
        use_columns: List[str] = ["temp", "hum", "windspeed", "mnth", "yr", "cnt"],
) -> tuple[pd.DataFrame, pd.Series]:
    # Download dataset
    url = "https://raw.githubusercontent.com/christophM/interpretable-ml-book/master/data/bike.csv"
    df = pd.read_csv(url)

    # Preprocess data
    df = df[use_columns]

    # Preprocess mnth column
    mnth_map = { "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12 }
    df["mnth"] = df["mnth"].map(mnth_map)

    X = df.drop("cnt", axis=1)
    y = df["cnt"]

    return X, y



def california_housing(
    use_columns: List[str] = [
        "MedInc", "HouseAge", "AveRooms", "AveBedrms",
        "Population", "AveOccup", "Latitude", "Longitude", "MedHouseVal"
    ],
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    California Housing (regression)
    Target: MedHouseVal (median house value)
    """
    data = fetch_california_housing(as_frame=True)
    df = data.frame

    # Restrict to chosen columns
    df = df[use_columns]

    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]
    return X, y


def diabetes(
    use_columns: List[str] = [
        "age", "sex", "bmi", "bp", "s1",
        "s2", "s3", "s4", "s5", "s6", "target"
    ],
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Diabetes dataset (regression)
    Target: progression of disease (target)
    """
    data = load_diabetes(as_frame=True)
    df = data.frame

    df = df[use_columns]

    X = df.drop("target", axis=1)
    y = df["target"]
    return X, y


def wine_quality_red(
    use_columns: List[str] = [
        "fixed acidity", "volatile acidity", "citric acid",
        "residual sugar", "chlorides", "free sulfur dioxide",
        "total sulfur dioxide", "density", "pH", "sulphates",
        "alcohol", "quality",
    ],
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Red wine quality (regression)
    Target: quality (integer score 0-10, often treated as regression)
    Source: UCI Wine Quality (red)
    """
    url = (
        "https://raw.githubusercontent.com/"
        "shrikant-temburwar/Wine-Quality-Dataset/master/winequality-red.csv"
    )
    # UCI version uses ; as separator
    df = pd.read_csv(url, sep=";")

    df = df[use_columns]

    X = df.drop("quality", axis=1)
    y = df["quality"]
    return X, y


def auto_mpg(
    use_columns: List[str] = [
        "cylinders", "displacement", "horsepower",
        "weight", "acceleration", "model-year", "mpg"
    ],
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Auto MPG (regression)
    Target: mpg (fuel efficiency)
    Source: UCI Auto MPG (cleaned CSV)
    """
    url = "https://raw.githubusercontent.com/plotly/datasets/master/auto-mpg.csv"
    # Handle '?' in horsepower as missing
    df = pd.read_csv(url, na_values=["?"])

    # Normalize column names a bit: "model year" -> "model_year", etc.
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    # Clean: drop rows with missing values
    df = df.dropna()

    # We don't want the car_name string column as a feature
    if "car_name" in df.columns:
        df = df.drop(columns=["car_name"])

    # Map requested use_columns (which use snake_case) + mpg
    df = df[use_columns]

    X = df.drop("mpg", axis=1)
    y = df["mpg"]
    return X, y


def toy_dataset(
        n_samples: int = 500,
        x_limits = [-math.pi, math.pi],
        rng: RandomGenerator = RandomGenerator(42),
        uniformity_x1: bool = True,
        uniformity_y2: bool = True,
        correlation: float = 0.0,
) -> tuple[pd.DataFrame, pd.Series]:

    low, high = x_limits

    # --- 1. Generate two lists of independent standard normals ---
    # rng.gauss(mean, std, n) returns a list of n values
    z1 = np.array(rng.gauss(0.0, 1.0, n_samples))
    z2 = np.array(rng.gauss(0.0, 1.0, n_samples))

    # --- 2. Impose correlation between z1 and z2 ---
    # If correlation = rho, construct:
    #   x2 = rho*z1 + sqrt(1-rho^2)*z2
    rho = float(correlation)
    if abs(rho) > 0.999:
        rho = 0.999 * math.copysign(1.0, rho)

    z2_corr = rho * z1 + math.sqrt(1 - rho**2) * z2

    # --- 3. Transform marginals ---
    def normal_to_uniform(z_vals, low, high):
        u = 0.5 * (1.0 + np.vectorize(math.erf)(z_vals / math.sqrt(2)))
        return low + (high - low) * u

    # x1
    if uniformity_x1:
        x1 = normal_to_uniform(z1, low, high)
    else:
        x1 = z1

    # x2
    if uniformity_y2:
        x2 = normal_to_uniform(z2_corr, low, high)
    else:
        x2 = z2_corr

    # --- 4. Nonlinear target function ---
    signal = np.sin(x1) + x2

    y = signal

    # --- 5. Package output ---
    X = pd.DataFrame({"x1": x1, "x2": x2})
    y_series = pd.Series(y, name="y")

    return X, y_series


# ---------------------------------------------------------------------------------------------------------------------
# Classification datasets
# ---------------------------------------------------------------------------------------------------------------------


def iris(
    use_columns: List[str] = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
        "species",
    ],
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Iris flower dataset (classification)
    Target: species (setosa, versicolor, virginica)
    """
    data = load_iris(as_frame=True)
    df = data.frame  # includes feature columns + 'target'

    # Map numeric target → string species name
    df["species"] = df["target"].map(lambda i: data.target_names[i])
    df = df.drop(columns=["target"])

    # Restrict to chosen columns
    df = df[use_columns]

    X = df.drop("species", axis=1)
    y = df["species"]
    return X, y


def penguins(
    use_columns: List[str] = [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
        "species",
    ],
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Palmer Penguins dataset (classification)
    Target: species (Adelie, Chinstrap, Gentoo)

    By default we only use numeric features for convenience.
    """
    url = (
        "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/"
        "master/inst/extdata/penguins.csv"
    )
    df = pd.read_csv(url)

    # Drop rows with missing values in selected columns
    df = df[use_columns].dropna()

    X = df.drop("species", axis=1)
    y = df["species"]
    return X, y



def breast_cancer(
    use_columns: List[str] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Breast Cancer Wisconsin dataset (binary classification)

    Target:
        0 = malignant
        1 = benign
    """

    data = load_breast_cancer(as_frame=True)
    df = data.frame  # includes all features + target column "target"

    # Map numeric target to string labels if desired
    df["label"] = df["target"].map({0: "malignant", 1: "benign"})
    df = df.drop(columns=["target"])

    # If no column selection provided, use all features + label
    if use_columns is None:
        use_columns = list(df.columns)  # all features + "label"

    df = df[use_columns]

    X = df.drop(columns=["label"])
    y = df["label"]

    return X, y
