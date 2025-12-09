from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.datasets import load_iris, load_wine
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score

from faex.resources.dataseting import bikes
from faex.core.ExplanationCore import ExplanationCore
from faex.core.DataCore import DataCore


# ---------- Data & model utilities ----------

DATASET_OPTIONS = {
    "Bike sharing": bikes,
}

MODEL_OPTIONS = {
    "Random Forest Regressor 20": RandomForestRegressor(n_estimators=20, random_state=42),
    "Random Forest Regressor 100": RandomForestRegressor(n_estimators=100, random_state=42),
    "Neural Network Regressor 5-5": MLPRegressor(max_iter=1000, hidden_layer_sizes=(5, 5)),
    "Neural Network Regressor 10-10": MLPRegressor(max_iter=1000, hidden_layer_sizes=(10, 10)),
}


def load_dataset(name: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Return (X, y, feature_names) for a given demo dataset."""
    return DATASET_OPTIONS[name]()  # type: ignore


def select_model(name: str):
    """Return an unfit sklearn model instance for the given name."""
    return MODEL_OPTIONS[name]  # type: ignore


def build_and_fit_model(model_name: str, X: pd.DataFrame, y: pd.Series) -> Tuple[Any, float, float]:
    """
    Instantiate and fit a simple sklearn model.
    Train it with cross validation of 5 folds.

    Returns the best fitted model, the R2 for training and test sets.
    """

    # Instantiate model
    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000, multi_class="auto")
    elif model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=200, random_state=42)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # 5-fold cross-validation on the full dataset (test R2 estimate)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="r2")

    # Fit final model on the full dataset (training R2)
    model.fit(X, y)
    y_pred_train = model.predict(X)
    r2_train = r2_score(y, y_pred_train)

    r2_test = float(np.mean(cv_scores))

    return model, r2_train, r2_test


def create_explaination_core(
        X: pd.DataFrame,
        model: Any,
        feature: str,
) -> ExplanationCore:
    """Create an ExplanationCore object for the given data and model."""
    datacore = DataCore(model, X, feature, bins=100)
    core = ExplanationCore(datacore)
    return core


# ---------- Per-tab state ----------

@dataclass
class TabState:
    # Display name (e.g. "Tab 1")
    name: str

    # Persistent objects
    dataset_name: Optional[str] = None
    model_name: Optional[str] = None
    feature_name: Optional[str] = None

    # Loaded data and fitted model
    core: ExplanationCore = None
    r2_train: float = None
    r2_test: float = None

    # Pure visualization parameters (safe to change any time)
    explainers: List[str]


# ---------- Session state helpers ----------

def create_initial_state():
    """Initialize session state with at least one tab."""
    if "tabs" not in st.session_state:
        st.session_state.tabs = {}        # tab_id -> TabState
        st.session_state.tab_order = []   # list of tab_ids for ordering
        st.session_state.next_tab_index = 1
        create_tab()


def create_tab(copy_from_id: Optional[str] = None) -> str:
    """Create a new tab. Optionally clone from an existing tab."""
    idx = st.session_state.next_tab_index
    st.session_state.next_tab_index += 1

    new_id = f"tab_{idx}"
    name = f"Tab {idx}"

    if copy_from_id and copy_from_id in st.session_state.tabs:
        src: TabState = st.session_state.tabs[copy_from_id]
        new_state = TabState(
            name=name,
            dataset_name=src.dataset_name,
            model_name=src.model_name,
            X=src.X,
            y=src.y,
            model=src.model,
            feature_names=list(src.feature_names),
            plot_params=dict(src.plot_params),
        )
    else:
        new_state = TabState(name=name)

    st.session_state.tabs[new_id] = new_state
    st.session_state.tab_order.append(new_id)
    return new_id


def delete_tab(tab_id: str):
    """Remove a tab and ensure at least one remains."""
    if tab_id in st.session_state.tabs:
        del st.session_state.tabs[tab_id]
        st.session_state.tab_order = [
            tid for tid in st.session_state.tab_order if tid != tab_id
        ]

    if not st.session_state.tab_order:
        create_tab()


def ensure_data_and_model(state: TabState, dataset_name: str, model_name: str, feature_name: str):
    """
    For this tab, load data and fit model ONLY when dataset or model changes.

    This ensures the (dataset, model) object is reused when you just tweak
    visualization parameters.
    """
    # Dataset
    if state.dataset_name != dataset_name or state.X is None or state.y is None:
        X, y, feature_names = load_dataset(dataset_name)
        state.dataset_name = dataset_name
        state.X = X
        state.y = y
        state.feature_names = feature_names
        state.model = None  # force model rebuild when dataset changes

    # Model
    if state.model_name != model_name or state.model is None:
        state.model = build_and_fit_model(model_name, state.X, state.y)
        state.model_name = model_name

    # Feature
    if state.feature_name != feature_name or state.feature_name is None:
        state.feature_name = feature_name
        state.core = create_explaination_core(state.X, state.model, feature_name)


# ---------- Plotting ----------

def make_plot(state: TabState):
    """Build the Plotly figure using persistent data/model + tab parameters."""
    X = state.X
    y = state.y

    if X is None or y is None:
        # Empty placeholder figure
        return px.scatter(title="No data loaded yet")

    params = state.plot_params
    feature_names = state.feature_names or list(X.columns)

    if len(feature_names) < 2:
        return px.scatter(title="Not enough features to plot a 2D scatter")

    # Resolve axis features with sane fallbacks
    x_default = params.get("x_feature", feature_names[0])
    if x_default not in feature_names:
        x_default = feature_names[0]

    y_default = params.get(
        "y_feature",
        feature_names[1] if len(feature_names) > 1 else feature_names[0],
    )
    if y_default not in feature_names:
        y_default = feature_names[1] if len(feature_names) > 1 else feature_names[0]

    x_feature = x_default
    y_feature = y_default

    color_mode = params.get("color_mode", "Target")  # "Target" or "Prediction"
    use_size = params.get("use_size", False)
    size_feature = params.get("size_feature")

    df = X.copy()
    df["target"] = y

    # Optional: color by prediction using the pre-fit model
    if color_mode == "Prediction" and state.model is not None:
        preds = state.model.predict(X)
        df["prediction"] = preds
        color_col = "prediction"
    else:
        color_col = "target"

    # Optional: active/inactive size parameter
    if use_size and size_feature in feature_names:
        size_col = size_feature
    else:
        size_col = None

    fig = px.scatter(
        df,
        x=x_feature,
        y=y_feature,
        color=color_col,
        size=size_col,
        hover_data=["target"],
        title=f"{state.dataset_name} - {state.model_name}",
    )

    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        legend_title=color_col.capitalize(),
    )
    return fig


# ---------- UI per tab ----------

def render_tab(tab_id: str, state: TabState):
    # Tab-level controls (duplicate / close)
    ctrl_cols = st.columns([1, 1, 6])
    with ctrl_cols[0]:
        if st.button("➕ Duplicate", key=f"dup_{tab_id}"):
            create_tab(copy_from_id=tab_id)
            st.experimental_rerun()  # or st.rerun() in newer Streamlit

    with ctrl_cols[1]:
        if st.button("✖ Close", key=f"close_{tab_id}"):
            delete_tab(tab_id)
            st.experimental_rerun()

    st.markdown("---")

    left_col, right_col = st.columns([1, 3])

    # ---- Left: configuration column ----
    with left_col:
        st.subheader("Configuration")

        # Dataset & model selection
        ds_index = (
            DATASET_OPTIONS.index(state.dataset_name)
            if state.dataset_name in DATASET_OPTIONS
            else 0
        )
        dataset_name = st.selectbox(
            "Dataset",
            DATASET_OPTIONS,
            index=ds_index,
            key=f"dataset_{tab_id}",
        )

        mdl_index = (
            MODEL_OPTIONS.index(state.model_name)
            if state.model_name in MODEL_OPTIONS
            else 0
        )
        model_name = st.selectbox(
            "Model",
            MODEL_OPTIONS,
            index=mdl_index,
            key=f"model_{tab_id}",
        )

        # Ensure persistent objects for this tab
        ensure_data_and_model(state, dataset_name, model_name)
        feature_names = state.feature_names or []

        st.markdown("### Parameters (template)")

        if feature_names:
            # Axis parameters
            x_default = state.plot_params.get("x_feature")
            if x_default not in feature_names:
                x_default = feature_names[0]

            y_default = state.plot_params.get("y_feature")
            if y_default not in feature_names:
                y_default = feature_names[1] if len(feature_names) > 1 else feature_names[0]

            state.plot_params["x_feature"] = st.selectbox(
                "X axis feature",
                feature_names,
                index=feature_names.index(x_default),
                key=f"x_feature_{tab_id}",
            )

            state.plot_params["y_feature"] = st.selectbox(
                "Y axis feature",
                feature_names,
                index=feature_names.index(y_default),
                key=f"y_feature_{tab_id}",
            )

            # Example: parameter that changes how we use the model
            state.plot_params["color_mode"] = st.radio(
                "Color by",
                options=["Target", "Prediction"],
                key=f"color_mode_{tab_id}",
            )

            # Example: list of parameters that can be active/inactive
            state.plot_params["use_size"] = st.checkbox(
                "Use a feature as point size",
                value=state.plot_params.get("use_size", False),
                key=f"use_size_{tab_id}",
            )

            size_disabled = not state.plot_params["use_size"]
            current_size_feature = state.plot_params.get("size_feature")
            if current_size_feature not in feature_names:
                current_size_feature = feature_names[0]

            state.plot_params["size_feature"] = st.selectbox(
                "Size feature",
                feature_names,
                index=feature_names.index(current_size_feature),
                key=f"size_feature_{tab_id}",
                disabled=size_disabled,
            )

            # You can extend this block with more parameters (sliders, switches, etc.)
            # They should only affect plotting, not dataset/model objects.
        else:
            st.info("Select a dataset to configure visualization parameters.")

    # ---- Right: central panel with Plotly figure ----
    with right_col:
        st.subheader("Visualization")
        fig = make_plot(state)
        st.plotly_chart(fig, use_container_width=True)


# ---------- Main app ----------

def main():
    st.set_page_config(page_title="Multi-tab Model Visualizer", layout="wide")

    st.title("Multi-tab Model Visualizer (template)")
    st.caption(
        "Each tab keeps its own dataset, fitted model and visualization parameters. "
        "Changing parameters only regenerates the Plotly figure - data/model objects are reused."
    )

    create_initial_state()

    # Global toolbar
    top_cols = st.columns([1, 7])
    with top_cols[0]:
        if st.button("➕ New tab", key="global_new_tab"):
            create_tab()
            st.experimental_rerun()

    # Create Streamlit tabs from our internal tab list
    tab_labels = [st.session_state.tabs[tid].name for tid in st.session_state.tab_order]
    st_tabs = st.tabs(tab_labels)

    # Render each tab independently
    for idx, tab_id in enumerate(st.session_state.tab_order):
        state: TabState = st.session_state.tabs[tab_id]
        with st_tabs[idx]:
            render_tab(tab_id, state)


if __name__ == "__main__":
    main()
