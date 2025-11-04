import logging
import sys
from pathlib import Path

logging.getLogger(__name__).addHandler(logging.NullHandler())


# Add src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Activate debugging
# Silence traces from other modules except my library
logger = logging.getLogger("faxai")   # use your package name here
logger.setLevel(logging.DEBUG)        # or INFO

########################################################################################################################

import numpy as np
import pandas as pd

from faxai.mathing.distribution.parametric_distributions import NormalDistribution, UniformDistribution
from faxai.mathing.distribution.UnionDistribution import UnionDistribution
from faxai.mathing.RandomGenerator import RandomGenerator

# Set pandas print options for better readability, wider column and not breaking lines
pd.set_option('display.precision', 4)
pd.set_option('display.width', 100)
pd.set_option('display.max_columns', None)


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
from faxai.explaining.DataCore import DataCore
from faxai.explaining.ExplainerConfiguration import ExplainerConfiguration
from faxai.explaining.explainers.ICE import ICE

core = ExplainerCore(
    dataframe_X=df_X,
    model=model
)

conf1 = ExplainerConfiguration(
    datacore=core.datacore(),
    study_features=["x1"],
    bins=5,
    strict_limits=False,
)

core.add_configuration("conf1", conf1)

ice = core.explain(technique="ice", configuration="conf1")


print(f"ICE Predictions: {ice}")

ice2 = core.explain(technique="ice", configuration="conf1")

ice2 = core.explain(technique="ice", configuration="conf1")

dataplotter = core.plot(technique="ice", configuration="conf1", params={"alpha":1.0})
fig, ax = dataplotter.matplotlib_plot()
fig.show()

print()
print(type(dataplotter))
print(dataplotter)
print(dataplotter.data)
for d in dataplotter.data:
    print(d.x)
    print(d.y)
    print(d.params)
print()



print(f"ICE Predictions: {ice}")

pdp = core.explain(technique="pdp", configuration="conf1")

print(f"PDP Predictions: {pdp}")

dataplotter = core.plot(technique="pdp", configuration="conf1")
fig, ax = dataplotter.matplotlib_plot()
fig.show()
