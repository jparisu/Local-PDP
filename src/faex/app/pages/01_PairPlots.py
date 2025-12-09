import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

import faex.resources.dataseting as dataseting


st.set_page_config(page_title="Dataset pairplots", layout="wide")

st.title("Dataset pairplots")

# Reuse the same datasets as in the main app
DATASET_OPTIONS = {
    "Bike sharing": dataseting.bikes(),
    "Toy dataset": dataseting.toy_dataset(),
    "Toy dataset - correlation": dataseting.toy_dataset(correlation=0.8),
    "California housing": dataseting.california_housing(),
    "Wine quality": dataseting.wine_quality_red(),
    "MPG dataset": dataseting.auto_mpg(),
    "Diabetes dataset": dataseting.diabetes(),

    "Iris dataset": dataseting.iris(),
    "Penguins": dataseting.penguins(),
    "Breast cancer": dataseting.breast_cancer(),
}

st.sidebar.header("Pairplot settings")

dataset_name = st.sidebar.selectbox(
    "Dataset",
    options=list(DATASET_OPTIONS.keys()),
)

max_rows = st.sidebar.slider(
    "Max rows (subsample for speed)",
    min_value=100,
    max_value=5000,
    value=1000,
    step=100,
)

hue_col = st.sidebar.text_input(
    "Hue column (optional, must exist in X or y)",
    value="target",
)

# Load selected dataset
df_X, df_y = DATASET_OPTIONS[dataset_name]

# Combine X and y into a single DataFrame for plotting
import pandas as pd

df = df_X.copy()
# Name the target column something simple if it's a Series
if df_y is not None:
    if isinstance(df_y, (pd.Series, pd.DataFrame)):
        if isinstance(df_y, pd.Series):
            df["target"] = df_y
        else:
            # If it's a DataFrame, just concat with suffix
            for col in df_y.columns:
                df[f"target_{col}"] = df_y[col]
    else:
        # Fallback: attempt to attach as 'target'
        df["target"] = df_y

# Subsample for speed
if len(df) > max_rows:
    df = df.sample(n=max_rows, random_state=42)

st.write(f"### Pairplot for: {dataset_name}")
st.write(f"Shape used for plot: {df.shape}")

with st.spinner("Generating pairplot..."):
    # Build kwargs for sns.pairplot
    pairplot_kwargs = {}
    if hue_col and hue_col in df.columns:
        pairplot_kwargs["hue"] = hue_col

    g = sns.pairplot(df, diag_kind="kde", **pairplot_kwargs)
    st.pyplot(g)
