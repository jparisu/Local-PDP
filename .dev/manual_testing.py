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

from faxai.mathing.distribution.parametric_distributions import NormalDistribution, UniformDistribution
from faxai.mathing.distribution.UnionDistribution import UnionDistribution
from faxai.mathing.RandomGenerator import RandomGenerator

n = NormalDistribution(mean=1.0, std=1.0)
u = UniformDistribution(low=-1.0, high=3.0)

mix = UnionDistribution([n, u])

print("Mixture Mean:", mix.mean())
print("Mixture Std Dev:", mix.std())

x = np.linspace(-5, 5, 11)

for i in x:
    print(f"x={i}:  PDF={mix.pdf(np.array([i]))[0]}, CDF={mix.cdf(np.array([i]))[0]}")
    print()
