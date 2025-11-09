import logging
import sys
from pathlib import Path

logging.getLogger(__name__).addHandler(logging.NullHandler())


# Add src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Activate debugging
# Silence traces from other modules except my library
logging.basicConfig(
    level=logging.DEBUG,
)

########################################################################################################################

import numpy as np
import pandas as pd

from faxai.mathing.distribution.parametric_distributions import NormalDistribution, UniformDistribution
from faxai.mathing.distribution.UnionDistribution import UnionDistribution
from faxai.mathing.RandomGenerator import RandomGenerator

# Set pandas print options for better readability, wider column and not breaking lines
pd.set_option('display.precision', 2)
pd.set_option('display.width', 100)
pd.set_option('display.max_columns', None)

# Set numpy print options for better readability
np.set_printoptions(precision=2, suppress=True)
np.set_printoptions(threshold=sys.maxsize)

rng = RandomGenerator(42)

N = 8
df = pd.DataFrame({
    "x1": np.linspace(0,10, N),
    "x2": rng.gauss(0, 1, N),
    "x3": np.linspace(0,10, N) + rng.gauss(0, 1, N),
    "target": np.linspace(0, 1, N)
})

df_X, df_y = df[["x1", "x2", "x3"]], df["target"]

print(f"Data: {df_X.head()}")


class MockModel:
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return 2 * df["x1"] + df["x3"]

model = MockModel()

print(f"Predictions: {model.predict(df_X.head())}")


from faxai.explaining.ExplainerCore import ExplainerCore
from faxai.explaining.DataCore import DataCore
from faxai.explaining.ExplainerConfiguration import ExplainerConfiguration
from faxai.explaining.explainers.ICE import ICE

core = ExplainerCore(
    dataframe_X=df_X,
    model=model
)

conf1 = ExplainerConfiguration(
    datacore=core.datacore(),
    study_features=["x1", "x2"],
    bins=10,
    strict_limits=False,
)


from faxai.explaining.explainers.L_ICE import L_ICE
from faxai.explaining.ExplainerFactory import GlobalExplainerFactory




print(f"CONFIGURATION: {conf1}")

core.add_configuration("conf1", conf1)


ice = core.explain(technique="ice", configuration="conf1")
print(ice)
