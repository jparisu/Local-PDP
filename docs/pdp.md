# Partial Dependence Plot (PDP)

Partial Dependence Plots (PDP) show the average effect of a feature on the model's predictions, marginalizing over all other features.

## Overview

PDP is a global interpretability method that shows how a feature affects predictions on average. It's computed by averaging Individual Conditional Expectation (ICE) curves across all instances.

## How PDP Works

For a selected feature, PDP:

1. Varies the feature across a range of values
2. For each value, makes predictions for all instances (with other features at their actual values)
3. Averages these predictions across all instances
4. Plots the average prediction as a function of the feature

The result is a single curve showing the average effect of the feature on the model's predictions.

## Basic Usage

Here's a simple example using a linear model:

```python
import numpy as np
import pandas as pd
from faxai.explaining.ExplainerCore import ExplainerCore
from faxai.explaining.ExplainerConfiguration import ExplainerConfiguration

# Create sample data
df_X = pd.DataFrame({
    "age": [25, 35, 45, 55, 65],
    "income": [30000, 50000, 70000, 90000, 110000],
    "credit_score": [600, 650, 700, 750, 800],
})

# Assume we have a trained model
# model = ... (your trained model)

# Create ExplainerCore
core = ExplainerCore(dataframe_X=df_X, model=model)

# Configure PDP for the 'age' feature
conf = ExplainerConfiguration(
    datacore=core.datacore(),
    study_features=["age"],
    bins=50,  # Number of points to evaluate
)

core.add_configuration("age_pdp", conf)

# Generate PDP explanation
pdp = core.explain(technique="pdp", configuration="age_pdp")

# pdp is a HyperPlane object with:
#   - pdp.grid: the grid of age values used
#   - pdp.target: average predictions at each grid point
```

## Plotting PDP

You can easily visualize PDP:

```python
# Create a plot
dataplotter = core.plot(technique="pdp", configuration="age_pdp")

# Render with matplotlib
fig, ax = dataplotter.matplotlib_plot()
fig.show()
```

The plot will show a single line representing the average effect of age on predictions.

## Understanding PDP Output

The PDP explainer returns a `HyperPlane` object with:

- **grid**: A `Grid` object containing the values used for the studied feature(s)
- **target**: A numpy array with shape `(n_grid_points,)` containing the average predictions

For example, if you evaluate 50 grid points:
- `pdp.target.shape` will be `(50,)`
- Each value represents the average prediction at that feature value

## PDP vs ICE

PDP is the average of all ICE curves. While ICE shows individual heterogeneity, PDP shows the overall trend:

- **ICE**: Shows how each individual instance responds to feature changes
- **PDP**: Shows the average response across all instances

You can visualize both together to see both the individual variation and the average trend:

```python
# Plot both ICE and PDP
ice_plotter = core.plot(technique="ice", configuration="age_pdp", params={"alpha": 0.2})
pdp_plotter = core.plot(technique="pdp", configuration="age_pdp")

# Combine plots
fig, ax = ice_plotter.matplotlib_plot()
pdp_plotter.matplotlib_plot(fig=fig, ax=ax)
ax.set_xlabel("age")
ax.set_ylabel("Prediction")
ax.set_title("ICE and PDP for age")
fig.show()
```

## When to Use PDP

PDP is particularly useful when:

- You want to understand the average effect of a feature
- You need a simple, interpretable summary of feature importance
- You want to communicate model behavior to non-technical stakeholders
- You need to validate that the model learns expected relationships

## Assumptions and Limitations

### Independence Assumption

PDP assumes that features are independent. When features are correlated, PDP may show unrealistic scenarios.

For example, if "house_size" and "number_of_rooms" are correlated, PDP might evaluate "small house with many rooms" which rarely occurs in reality.

### Heterogeneity

PDP averages over all instances, which can hide important heterogeneity. If different groups of instances have opposite trends, PDP might show a flat line even though the feature is important.

**Solution**: Use ICE plots alongside PDP to see individual variation.

## Advanced Configuration

### Custom Feature Values

Specify exact feature values to evaluate:

```python
conf = ExplainerConfiguration(
    datacore=core.datacore(),
    study_features=["age"],
    feature_values={"age": np.array([20, 30, 40, 50, 60, 70, 80])},
    feature_limits={"age": (20, 80)},
    use_default=False,
)
```

### Feature Limits

Control the range of feature values:

```python
conf = ExplainerConfiguration(
    datacore=core.datacore(),
    study_features=["age"],
    bins=50,
    strict_limits=True,  # Use exact data min/max
)

# Or extend beyond data range
conf = ExplainerConfiguration(
    datacore=core.datacore(),
    study_features=["age"],
    bins=50,
    strict_limits=False,  # Extends by half a bin width
)
```

## Example: Complete Workflow

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from faxai.explaining.ExplainerCore import ExplainerCore
from faxai.explaining.ExplainerConfiguration import ExplainerConfiguration

# Create sample data
np.random.seed(42)
n_samples = 200

df_X = pd.DataFrame({
    "feature1": np.random.uniform(0, 10, n_samples),
    "feature2": np.random.uniform(0, 5, n_samples),
    "feature3": np.random.uniform(-2, 2, n_samples),
})

# Create target with a known relationship
# feature1 has a non-linear effect
y = (
    2 * df_X["feature1"] 
    + 0.5 * df_X["feature1"]**2  # Non-linear term
    + 3 * df_X["feature2"] 
    + np.random.normal(0, 1, n_samples)
)

# Train a model
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(df_X, y)

# Create ExplainerCore
core = ExplainerCore(dataframe_X=df_X, model=model)

# Configure for feature1
conf = ExplainerConfiguration(
    datacore=core.datacore(),
    study_features=["feature1"],
    bins=50,
)

core.add_configuration("feature1_pdp", conf)

# Generate and plot PDP
pdp = core.explain(technique="pdp", configuration="feature1_pdp")
dataplotter = core.plot(technique="pdp", configuration="feature1_pdp")
fig, ax = dataplotter.matplotlib_plot()
ax.set_xlabel("feature1")
ax.set_ylabel("Average Prediction")
ax.set_title("PDP for feature1")
ax.grid(True, alpha=0.3)
fig.show()

# The plot should show the non-linear quadratic relationship
```

## Combining ICE and PDP

The most powerful approach is to use ICE and PDP together:

```python
# Single configuration for both
conf = ExplainerConfiguration(
    datacore=core.datacore(),
    study_features=["feature1"],
    bins=50,
)

core.add_configuration("combined", conf)

# Plot ICE curves (semi-transparent)
ice_plotter = core.plot(
    technique="ice", 
    configuration="combined",
    params={"alpha": 0.1, "color": "lightblue"}
)

fig, ax = ice_plotter.matplotlib_plot()

# Overlay PDP (bold line)
pdp_plotter = core.plot(
    technique="pdp",
    configuration="combined",
    params={"linewidth": 3, "color": "darkblue"}
)

pdp_plotter.matplotlib_plot(fig=fig, ax=ax)
ax.set_xlabel("feature1")
ax.set_ylabel("Prediction")
ax.set_title("ICE (light blue) and PDP (dark blue) for feature1")
ax.legend()
fig.show()
```

## API Reference

::: faxai.explaining.explainers.PDP
    options:
      show_source: true
