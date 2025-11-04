# Getting Started

This guide will help you get started with `faxai` for model explainability.

## Installation

First, install the library: [Installation](installation.md)

## Your First Explanation

Let's walk through a complete example of using `faxai` to explain a machine learning model.

### Step 1: Prepare Your Data and Model

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Create sample data
np.random.seed(42)
n_samples = 100

df_X = pd.DataFrame({
    "age": np.random.uniform(20, 70, n_samples),
    "income": np.random.uniform(20000, 120000, n_samples),
    "credit_score": np.random.uniform(500, 850, n_samples),
})

# Create target with known relationships
y = (
    1000 * (df_X["age"] / 50)  # Age effect
    + 0.3 * df_X["income"]      # Income effect
    + 50 * (df_X["credit_score"] / 700)  # Credit score effect
    + np.random.normal(0, 1000, n_samples)  # Noise
)

# Train a model
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(df_X, y)
```

### Step 2: Create ExplainerCore

```python
from faxai.explaining.ExplainerCore import ExplainerCore

# Create the explainer
core = ExplainerCore(dataframe_X=df_X, model=model)
```

### Step 3: Configure Your Explanation

```python
from faxai.explaining.ExplainerConfiguration import ExplainerConfiguration

# Configure to explain the effect of age
conf = ExplainerConfiguration(
    datacore=core.datacore(),
    study_features=["age"],  # Feature to study
    bins=50,                  # Number of evaluation points
)

# Add configuration to the core
core.add_configuration("age_analysis", conf)
```

### Step 4: Generate ICE Explanation

```python
# Generate Individual Conditional Expectation
ice = core.explain(technique="ice", configuration="age_analysis")

print(f"ICE Results:")
print(f"  Grid shape: {ice.grid.shape()}")
print(f"  Predictions shape: {ice.targets.shape}")
print(f"  Number of instances: {ice.targets.shape[1]}")
print(f"  Number of grid points: {ice.targets.shape[0]}")
```

### Step 5: Generate PDP Explanation

```python
# Generate Partial Dependence Plot
pdp = core.explain(technique="pdp", configuration="age_analysis")

print(f"\nPDP Results:")
print(f"  Grid shape: {pdp.grid.shape()}")
print(f"  Predictions shape: {pdp.target.shape}")
```

### Step 6: Visualize Results

```python
# Plot ICE curves
ice_plotter = core.plot(
    technique="ice",
    configuration="age_analysis",
    params={"alpha": 0.2, "color": "lightblue"}
)

fig, ax = ice_plotter.matplotlib_plot()

# Overlay PDP
pdp_plotter = core.plot(
    technique="pdp",
    configuration="age_analysis",
    params={"linewidth": 3, "color": "darkblue", "label": "Average (PDP)"}
)

pdp_plotter.matplotlib_plot(fig=fig, ax=ax)

# Customize plot
ax.set_xlabel("Age (years)")
ax.set_ylabel("Predicted Value")
ax.set_title("Effect of Age on Model Predictions")
ax.legend()
ax.grid(True, alpha=0.3)
fig.show()
```

## Understanding the Results

### ICE (Individual Conditional Expectation)

- Shows how each individual instance's prediction changes as age varies
- Each light blue line represents one instance
- Reveals heterogeneity: different instances may respond differently to age changes
- Shape: `(n_grid_points, n_instances)`

### PDP (Partial Dependence Plot)

- Shows the average effect of age across all instances
- The dark blue line is the average of all ICE curves
- Provides a global view of the feature's impact
- Shape: `(n_grid_points,)`

## Next Steps

Now that you understand the basics, explore:

- **[ICE Documentation](ice.md)**: Detailed guide on Individual Conditional Expectation
- **[PDP Documentation](pdp.md)**: Comprehensive PDP usage and interpretation
- **Advanced Configurations**: Custom feature ranges, multiple features, and more

## Common Patterns

### Explaining Multiple Features

```python
# Study income instead of age
conf_income = ExplainerConfiguration(
    datacore=core.datacore(),
    study_features=["income"],
    bins=50,
)

core.add_configuration("income_analysis", conf_income)

# Generate explanations
ice_income = core.explain(technique="ice", configuration="income_analysis")
pdp_income = core.explain(technique="pdp", configuration="income_analysis")
```

### Custom Feature Ranges

```python
# Specify exact age values to study
conf_custom = ExplainerConfiguration(
    datacore=core.datacore(),
    study_features=["age"],
    feature_values={"age": np.array([25, 35, 45, 55, 65])},
    feature_limits={"age": (25, 65)},
    use_default=False,
)

core.add_configuration("custom_age", conf_custom)
```

### Comparing Different Features

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

features = ["age", "income", "credit_score"]
for idx, feature in enumerate(features):
    conf = ExplainerConfiguration(
        datacore=core.datacore(),
        study_features=[feature],
        bins=30,
    )
    core.add_configuration(f"{feature}_compare", conf)
    
    # Plot PDP for each feature
    pdp_plotter = core.plot(
        technique="pdp",
        configuration=f"{feature}_compare"
    )
    pdp_plotter.matplotlib_plot(fig=fig, ax=axes[idx])
    axes[idx].set_xlabel(feature.replace("_", " ").title())
    axes[idx].set_ylabel("Prediction")
    axes[idx].set_title(f"PDP: {feature}")
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
fig.show()
```

## Tips for Effective Use

1. **Start with PDP**: Get a global view before diving into individual instances
2. **Check ICE for heterogeneity**: If ICE curves vary widely, be cautious about PDP interpretations
3. **Use appropriate bins**: More bins give smoother curves but take longer to compute
4. **Consider feature ranges**: Use `strict_limits=True` to stay within observed data
5. **Combine with domain knowledge**: Validate that results align with expectations

## Troubleshooting

### Kernel Creation Issues

If you see errors about bandwidth or kernel creation with small datasets:

```python
# Disable automatic kernel creation
conf = ExplainerConfiguration(
    datacore=core.datacore(),
    study_features=["age"],
    feature_values={"age": np.linspace(20, 70, 30)},
    feature_limits={"age": (20, 70)},
    use_default=False,  # Skip automatic defaults
)
```

### Memory Issues with Large Datasets

For large datasets, use fewer bins or sample your data:

```python
# Use fewer evaluation points
conf = ExplainerConfiguration(
    datacore=core.datacore(),
    study_features=["age"],
    bins=20,  # Reduced from 50
)

# Or sample your data
sample_indices = np.random.choice(len(df_X), size=1000, replace=False)
df_X_sample = df_X.iloc[sample_indices]
core_sample = ExplainerCore(dataframe_X=df_X_sample, model=model)
```

## Learn More

- [ICE Documentation](ice.md) - Detailed ICE guide
- [PDP Documentation](pdp.md) - Comprehensive PDP reference
- [GitHub Examples](https://github.com/jparisu/faxai/tree/main/.dev) - More code examples
