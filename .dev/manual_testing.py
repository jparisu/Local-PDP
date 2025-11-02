import logging
import sys
from pathlib import Path

logging.getLogger(__name__).addHandler(logging.NullHandler())


# Add src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

logging.basicConfig(level=logging.DEBUG)

########################################################################################################################

import numpy as np
import pandas as pd

from faxai.mathing.distribution.parametric_distributions import NormalDistribution, UniformDistribution
from faxai.mathing.distribution.UnionDistribution import UnionDistribution
from faxai.mathing.RandomGenerator import RandomGenerator

rng = RandomGenerator()

N = 5
df = pd.DataFrame({
    "x1": np.linspace(0, 1, N),
    "x2": np.array(range(N,0,-1)),
    "x3": rng.gauss(0, 1, N),
    "target": np.linspace(5, 15, N)
})


df_X, df_y = df[["x1", "x2", "x3"]], df["target"]

print(f"Data: {df_X.head()}")

class MockModel:
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return df["x1"] * 10 + 5 + df["x3"]

model = MockModel()

print(f"Predictions: {model.predict(df_X.head())}")


from faxai.explaining.ExplainerCore import ExplainerCore
from faxai.explaining.configuration.DataCore import DataCore
from faxai.explaining.configuration.ExplainerConfiguration import ExplainerConfiguration
from faxai.explaining.ICE import ICE

datacore = DataCore(model=model, df_X=df_X)
core = ExplainerCore(datacore=datacore)

conf1 = ExplainerConfiguration(
    datacore=datacore,
    study_features=["x1","x2"],
    bins=3,
    strict_limits=False,
)

core.add_configuration("conf1", conf1)

ice = core.explain(ICE, configuration="conf1")

# Set pandas print options for better readability, wider column and not breaking lines
pd.set_option('display.precision', 4)
pd.set_option('display.width', 100)
pd.set_option('display.max_columns', None)


print(f"ICE Predictions: {ice}")

ice2 = core.explain(ICE, configuration="conf1")


print(f"ICE Predictions: {ice}")
