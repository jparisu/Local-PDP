# Individual Conditional Expectation (ICE)

Individual Conditional Expectation (ICE) plots show how a model's predictions change for each individual instance as a feature varies across its range.

## Overview

ICE plots are a powerful tool for understanding how individual predictions respond to changes in feature values. Unlike global methods that average over all instances, ICE shows the heterogeneity in the model's predictions across different instances.

## How ICE Works

For each instance in your dataset, ICE:

1. Varies a selected feature (or features) across a range of values
2. Keeps all other features at their original values for that instance
3. Makes predictions for each variation
4. Plots the prediction as a function of the varied feature

This creates one line per instance, showing how that specific instance's prediction changes as the feature varies.

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

# Configure ICE for the 'age' feature
conf = ExplainerConfiguration(
    datacore=core.datacore(),
    study_features=["age"],
    bins=50,  # Number of points to evaluate
)

core.add_configuration("age_ice", conf)

# Generate ICE explanation
ice = core.explain(technique="ice", configuration="age_ice")

# ice is a HyperPlanes object with:
#   - ice.grid: the grid of age values used
#   - ice.targets: predictions for each (grid_point, instance)
```

## Plotting ICE

You can easily visualize ICE plots:

```python
# Create a plot
dataplotter = core.plot(technique="ice", configuration="age_ice", params={"alpha": 0.3})

# Render with matplotlib
fig, ax = dataplotter.matplotlib_plot()
fig.show()
```

The plot will show one line per instance, each representing how that instance's prediction changes as age varies.

## Understanding ICE Output

The ICE explainer returns a `HyperPlanes` object with:

- **grid**: A `Grid` object containing the values used for the studied feature(s)
- **targets**: A numpy array with shape `(n_grid_points, n_instances)` containing the predictions

For example, if you have 5 instances and evaluate 50 grid points:
- `ice.targets.shape` will be `(50, 5)`
- Each column represents one instance's ICE curve
- Each row represents predictions at a specific feature value

## Advanced Configuration

### Custom Feature Values

Instead of using automatic binning, you can specify exact feature values:

```python
conf = ExplainerConfiguration(
    datacore=core.datacore(),
    study_features=["age"],
    feature_values={"age": np.array([20, 30, 40, 50, 60, 70, 80])},
    feature_limits={"age": (20, 80)},
    use_default=False,
)
```

### Multiple Features

You can study multiple features simultaneously (though visualization becomes more complex):

```python
conf = ExplainerConfiguration(
    datacore=core.datacore(),
    study_features=["age", "income"],
    bins=20,
)
```

## When to Use ICE

ICE plots are particularly useful when:

- You want to understand prediction variability across instances
- You suspect the model behaves differently for different types of instances
- You need to validate that the model's behavior aligns with domain knowledge
- You want to identify instances with unusual or unexpected behavior

## Relationship to PDP

ICE plots are closely related to Partial Dependence Plots (PDP). In fact, a PDP is simply the average of all ICE curves. While ICE shows individual heterogeneity, PDP shows the average effect.

## Example: Complete Workflow

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from faxai.explaining.ExplainerCore import ExplainerCore
from faxai.explaining.ExplainerConfiguration import ExplainerConfiguration

# Create sample data
np.random.seed(42)
n_samples = 100

df_X = pd.DataFrame({
    "feature1": np.random.uniform(0, 10, n_samples),
    "feature2": np.random.uniform(0, 5, n_samples),
    "feature3": np.random.uniform(-2, 2, n_samples),
})

# Create target with a known relationship
y = 2 * df_X["feature1"] + 3 * df_X["feature2"] + np.random.normal(0, 1, n_samples)

# Train a model
model = RandomForestRegressor(n_estimators=10, random_state=42)
model.fit(df_X, y)

# Create ExplainerCore
core = ExplainerCore(dataframe_X=df_X, model=model)

# Configure for feature1
conf = ExplainerConfiguration(
    datacore=core.datacore(),
    study_features=["feature1"],
    bins=30,
)

core.add_configuration("feature1_ice", conf)

# Generate and plot ICE
ice = core.explain(technique="ice", configuration="feature1_ice")
dataplotter = core.plot(technique="ice", configuration="feature1_ice")
fig, ax = dataplotter.matplotlib_plot()
ax.set_xlabel("feature1")
ax.set_ylabel("Prediction")
ax.set_title("ICE Plot for feature1")
fig.show()
```

## API Reference

::: faxai.explaining.explainers.ICE
    options:
      show_source: true
