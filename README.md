# Feature Attribution Explainability Python Library

[![Docs](https://readthedocs.org/projects/faxai/badge/?version=latest)](https://faxai.readthedocs.io/en/latest/?badge=latest)
[![CI](https://github.com/jparisu/faxai/actions/workflows/ci.yml/badge.svg)](https://github.com/jparisu/faxai/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/jparisu/faxai/branch/main/graph/badge.svg)](https://codecov.io/gh/jparisu/faxai)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/jparisu/faxai/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)


Feature Attribution Model-Agnostic **Python library** for Machine Learning **Explainability**.

This repository implements `faxai` python library with several feature attribution agnostic-model methods for explaining predictions of machine learning models.
Mainly, it focuses on **Local-PDP** (Local Partial Dependence Plots) method, which is designed to provide reliable explanations in any dataset, removing PDP independence assumptions.

📘 **Documentation:** [https://faxai.readthedocs.io](https://faxai.readthedocs.io)

## Features

- **ICE (Individual Conditional Expectation)**: Understand how predictions change for individual instances
- **PDP (Partial Dependence Plot)**: Visualize the average effect of features on predictions
- **Local-PDP**: Advanced method for reliable feature attribution without independence assumptions
- **Model-Agnostic**: Works with any machine learning model
- **Easy to Use**: Simple API for quick integration

## Quick Start

### Installation

```bash
pip install git+https://github.com/jparisu/faxai.git
```

### Basic Example

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from faxai.explaining.ExplainerCore import ExplainerCore
from faxai.explaining.ExplainerConfiguration import ExplainerConfiguration

# Prepare your data and model
df_X = pd.DataFrame({
    "age": [25, 35, 45, 55, 65],
    "income": [30000, 50000, 70000, 90000, 110000],
    "credit_score": [600, 650, 700, 750, 800],
})

y = np.array([20000, 35000, 50000, 65000, 80000])

# Train a model
model = RandomForestRegressor(random_state=42)
model.fit(df_X, y)

# Create ExplainerCore
core = ExplainerCore(dataframe_X=df_X, model=model)

# Configure explanation
conf = ExplainerConfiguration(
    datacore=core.datacore(),
    study_features=["age"],  # Feature to explain
    bins=50,
)

core.add_configuration("age_analysis", conf)

# Generate ICE (Individual Conditional Expectation)
ice = core.explain(technique="ice", configuration="age_analysis")
print(f"ICE shape: {ice.targets.shape}")  # (n_grid_points, n_instances)

# Generate PDP (Partial Dependence Plot)
pdp = core.explain(technique="pdp", configuration="age_analysis")
print(f"PDP shape: {pdp.target.shape}")  # (n_grid_points,)

# Visualize
dataplotter = core.plot(technique="pdp", configuration="age_analysis")
fig, ax = dataplotter.matplotlib_plot()
ax.set_xlabel("Age")
ax.set_ylabel("Prediction")
ax.set_title("PDP: Effect of Age on Predictions")
fig.show()
```

## ICE: Individual Conditional Expectation

ICE plots show how individual predictions change as a feature varies:

```python
# Configure for ICE
conf = ExplainerConfiguration(
    datacore=core.datacore(),
    study_features=["income"],
    bins=30,
)

core.add_configuration("income_ice", conf)

# Generate and plot ICE
ice_plotter = core.plot(
    technique="ice",
    configuration="income_ice",
    params={"alpha": 0.3}  # Semi-transparent lines
)

fig, ax = ice_plotter.matplotlib_plot()
ax.set_xlabel("Income")
ax.set_ylabel("Prediction")
ax.set_title("ICE: Individual Effects of Income")
fig.show()
```

Each line represents one instance's prediction trajectory as income varies. This reveals heterogeneity in how the model treats different instances.

## PDP: Partial Dependence Plot

PDP shows the average effect of a feature:

```python
# Generate and plot PDP
pdp_plotter = core.plot(technique="pdp", configuration="income_ice")
fig, ax = pdp_plotter.matplotlib_plot()
ax.set_xlabel("Income")
ax.set_ylabel("Average Prediction")
ax.set_title("PDP: Average Effect of Income")
ax.grid(True, alpha=0.3)
fig.show()
```

PDP is the average of all ICE curves, showing the global trend.

## Combining ICE and PDP

The most powerful approach is to visualize both together:

```python
# Plot ICE curves (transparent)
ice_plotter = core.plot(
    technique="ice",
    configuration="income_ice",
    params={"alpha": 0.15, "color": "lightblue"}
)

fig, ax = ice_plotter.matplotlib_plot()

# Overlay PDP (bold)
pdp_plotter = core.plot(
    technique="pdp",
    configuration="income_ice",
    params={"linewidth": 3, "color": "darkblue", "label": "Average (PDP)"}
)

pdp_plotter.matplotlib_plot(fig=fig, ax=ax)
ax.set_xlabel("Income")
ax.set_ylabel("Prediction")
ax.set_title("ICE and PDP: Individual and Average Effects")
ax.legend()
fig.show()
```

This reveals both individual variation (ICE) and overall trends (PDP).

## Advanced Usage

### Custom Feature Values

```python
# Specify exact values to evaluate
conf = ExplainerConfiguration(
    datacore=core.datacore(),
    study_features=["age"],
    feature_values={"age": np.array([20, 30, 40, 50, 60, 70, 80])},
    feature_limits={"age": (20, 80)},
    use_default=False,
)
```

### Multiple Features

```python
# Study interaction between features
conf = ExplainerConfiguration(
    datacore=core.datacore(),
    study_features=["age", "income"],
    bins=20,
)
```

## Documentation

For detailed documentation, examples, and API reference, visit:

📘 [https://faxai.readthedocs.io](https://faxai.readthedocs.io)

- [ICE Documentation](https://faxai.readthedocs.io/en/latest/ice/)
- [PDP Documentation](https://faxai.readthedocs.io/en/latest/pdp/)
- [Getting Started Guide](https://faxai.readthedocs.io/en/latest/getting-started/)

## Contributing

We welcome contributions! Please check our [Contributing Guide](https://github.com/jparisu/faxai/blob/main/CONTRIBUTING.md).

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](https://github.com/jparisu/faxai/blob/main/LICENSE) file for details.

## Citation

If you use this library in your research, please cite it as described in [CITATION.cff](https://github.com/jparisu/faxai/blob/main/CITATION.cff).
