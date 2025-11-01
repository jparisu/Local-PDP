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

from faxai.mathing.kernel import Bandwidth

matrix = np.array([[0.2, 0.01], [0.01, 0.5]])

if np.any(matrix <= 0):
    logging.error("Bandwidth matrix must have all positive values.")

try:
    print(Bandwidth.reckon_silverman_bandwidth(samples=100, sigma=np.array([1.0, 2.0])))
except Exception as e:
    logging.error("Error computing Silverman bandwidth: %s", e)
