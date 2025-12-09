import logging
import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Silence traces from other modules except my library
logging.basicConfig(level=logging.WARNING)

# Activate debugging
logging.getLogger("faex").setLevel(logging.DEBUG)

########################################################################################################################
import pandas as pd
import numpy as np

TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 30

# Download dataset
url = "https://raw.githubusercontent.com/christophM/interpretable-ml-book/master/data/bike.csv"
df = pd.read_csv(url)

# Preprocess data
columns = ["temp", "hum", "windspeed", "mnth", "yr", "cnt"]
df = df[columns]

# Preprocess mnth column
mnth_map = { "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12 }
df["mnth"] = df["mnth"].map(mnth_map)

# Show dataset
df.head()


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

df_x = df.drop(columns=["cnt"])
df_y = df["cnt"]

# Divide in train and test set
X_train, X_test, y_train, y_test = train_test_split(
    df_x,
    df_y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

# Train model
model = RandomForestRegressor(random_state=RANDOM_STATE, n_estimators=30)
model.fit(X_train, y_train)

# Evaluate model
r2_train = model.score(X_train, y_train)
r2_test = model.score(X_test, y_test)

print(f"R2 on train set: {r2_train:.3f}")
print(f"R2 on test set: {r2_test:.3f}")


from faex.core.DataCore import DataCore
from faex.core.ExplanationCore import ExplanationCore

FEATURE = "temp"

# Generate DataCore
datacore = DataCore(
        df_X=df_x,
        model=model,
        study_features=[FEATURE],
)

# Generate explainer
explainer = ExplanationCore(datacore=datacore)


from faex.explaining.ExplainerFactory import GlobalExplainerFactory, ExplainerFactory
from faex.explaining.explainers.ICE import ICE
factory = GlobalExplainerFactory()
print(factory)
print(type(factory))

# ExplainerFactory.register_explainer(
#     explainer=ICE,
#     aliases=["ice"],
# )
print(factory.get_available_explainers())


# Explaining PDP
explainer.visualize_doubleplot(
    explanations=[
        "real-prediction",
        "histogram",
        "distribution",
        "ice",
        "pdp",
        "pdp-d",
    ],
    matplotlib=True,
)

# Explaining l-PDP
explainer.visualize_doubleplot(
    explanations=[
        "real-prediction",
        "histogram",
        "distribution",
        "KernelNormalizer",
        # "l-ice",
        "l-pdp",
        "l-pdp-d",
    ],
    matplotlib=True,
)

import plotly.graph_objects as go

USE_MATPLOTLIB = False

# Explaining PDP
explainer.visualize_doubleplot(
    explanations=[
        "real-prediction",
        "histogram",
        "distribution",
        "ice",
        "pdp",
    ],
    matplotlib=USE_MATPLOTLIB,
)
