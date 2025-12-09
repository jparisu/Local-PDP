"""Streamlit app for managing multiple ExplanationCore-based tabs.

Each tab holds an ExplanationCore instance and a set of active explainers.
Tabs can be created, duplicated (sharing the same core object), or deleted.

You must provide:
- DATASET_OPTIONS: mapping from human-readable dataset name to dataset object
- MODEL_OPTIONS: mapping from human-readable model name to a fitted/ready model
- ExplanationCore implementation and its constructor signature

Assumed ExplanationCore signature (adapt as needed):
    ExplanationCore(dataset, model, feature_name, dataset_name=None, model_name=None)

Assumed visualization method (Plotly):
    core.visualize_doubleplot_plotly(explanations: list[str]) -> plotly.graph_objs.Figure
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List
import numpy as np

import streamlit as st

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score

from faex.core.DataCore import DataCore
from faex.core.ExplanationCore import ExplanationCore
from faex.explaining.ExplainerFactory import GlobalExplainerFactory
import faex.resources.dataseting as dataseting


# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

DATASET_OPTIONS = {
    "Bike sharing": dataseting.bikes(),
    "Toy dataset": dataseting.toy_dataset(),
    "California housing": dataseting.california_housing(),
    "Wine quality": dataseting.wine_quality_red(),
    "MPG dataset": dataseting.auto_mpg(),
    "Diabetes dataset": dataseting.diabetes(),
}

MODEL_OPTIONS = {
    "Random Forest Regressor 20": RandomForestRegressor(n_estimators=20, random_state=42),
    "Random Forest Regressor 100": RandomForestRegressor(n_estimators=100, random_state=42),
    "Neural Network Regressor 5-5": MLPRegressor(max_iter=1000, hidden_layer_sizes=(5, 5)),
    "Neural Network Regressor 10-10": MLPRegressor(max_iter=1000, hidden_layer_sizes=(10, 10)),
    "Linear Regression": LinearRegression(),
    "Support Vector Regressor": SVR(),
}

EXPLAINERS: List[str] = GlobalExplainerFactory().get_available_plot_explainers()
DEFAULT_EXPLAINERS: List[str] = ["realprediction", "histogram"]

# -----------------------------------------------------------------------------
# STATE STRUCTURES
# -----------------------------------------------------------------------------


@dataclass
class TabState:
    """Holds the state for one logical tab.

    core: ExplanationCore instance
    active_explainers: list of explainer names currently enabled in this tab
    """

    name: str
    core: Any  # Use "ExplanationCore" if you import the actual class
    active_explainers: List[str] = field(default_factory=list)


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------

def get_app_config() -> Dict[str, Any]:
    """Get or initialize global app configuration stored in session_state."""
    if "plot_backend" not in st.session_state:
        st.session_state.plot_backend = "Matplotlib"  # or "Matplotlib" as default

    if "bins" not in st.session_state:
        st.session_state.bins = 100

    if "kernel_factor" not in st.session_state:
        st.session_state.kernel_factor = 1.0

    if "data_percentage" not in st.session_state:
        st.session_state.data_percentage = 1.0

    return {
        "plot_backend": st.session_state.plot_backend,
        "bins": st.session_state.bins,
        "kernel_factor": st.session_state.kernel_factor,
        "data_percentage": st.session_state.data_percentage,
    }


def init_session_state() -> None:
    """Initialize the Streamlit session_state structures if needed."""

    if "tabs" not in st.session_state:
        # Mapping: tab_id -> TabState
        st.session_state.tabs = {}

    if "next_tab_id" not in st.session_state:
        st.session_state.next_tab_id = 1

    if "active_tab_id" not in st.session_state:
        st.session_state.active_tab_id = None


def get_feature_names(dataset: Any) -> List[str]:
    """Infer feature names from a dataset object."""

    # pandas-like
    if hasattr(dataset, "columns"):
        try:
            return list(dataset.columns)
        except Exception:
            pass

    # dict-like
    if isinstance(dataset, dict):
        return list(dataset.keys())

    # Fallback dummy values
    return ["feature1", "feature2", "feature3"]


def train_model(model: Any, X: Any, y: Any) -> Any:
    """Fit model with 5-fold CV and return trained model and scores."""

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="r2")

    model.fit(X, y)
    return model, np.mean(cv_scores)

def create_core(dataset_key: str, model_key: str, feature_name: str) -> Any:
    """Factory for ExplanationCore."""

    cfg = get_app_config()

    df_x, df_y = DATASET_OPTIONS[dataset_key]
    model_obj = MODEL_OPTIONS[model_key]

    model, score = train_model(model_obj, df_x, df_y)

    datacore = DataCore(
        model=model,
        df_X=df_x,
        study_features=[feature_name],
        bins=cfg["bins"],
        locality_factor=cfg["kernel_factor"],
        data_percentage=cfg["data_percentage"],
    )

    return ExplanationCore(datacore), score


def add_new_tab(tab_name: str, dataset_key: str, model_key: str, feature_name: str) -> None:
    """Create a new tab with its own ExplanationCore instance."""

    core, score = create_core(dataset_key, model_key, feature_name)
    tab_id = str(st.session_state.next_tab_id)
    st.session_state.next_tab_id += 1

    st.session_state.tabs[tab_id] = TabState(
        name=tab_name,
        core=core,
        active_explainers=DEFAULT_EXPLAINERS.copy(),
    )
    st.session_state.active_tab_id = tab_id


def duplicate_tab(tab_id: str) -> None:
    """Duplicate a tab while sharing the same core object."""

    original = st.session_state.tabs[tab_id]
    new_id = str(st.session_state.next_tab_id)
    st.session_state.next_tab_id += 1

    st.session_state.tabs[new_id] = TabState(
        name=f"{original.name} (copy)",
        core=original.core,  # shared reference
        active_explainers=original.active_explainers.copy(),
    )
    st.session_state.active_tab_id = new_id


def delete_tab(tab_id: str) -> None:
    """Delete a tab by its id."""

    if tab_id in st.session_state.tabs:
        del st.session_state.tabs[tab_id]

    if st.session_state.active_tab_id == tab_id:
        if st.session_state.tabs:
            st.session_state.active_tab_id = next(iter(st.session_state.tabs.keys()))
        else:
            st.session_state.active_tab_id = None


# -----------------------------------------------------------------------------
# UI SECTIONS
# -----------------------------------------------------------------------------

def render_sidebar() -> None:
    """Sidebar with controls to create new tabs."""

    st.sidebar.header("Tab manager")

    if not DATASET_OPTIONS or not MODEL_OPTIONS:
        st.sidebar.warning(
            "Define DATASET_OPTIONS and MODEL_OPTIONS in the code to enable tab creation."
        )
        return

    cfg = get_app_config()
    with st.sidebar.expander("Create new tab", expanded=True):

        # 1. Plot backend
        st.session_state.plot_backend = st.radio(
            "Plot backend",
            options=["Plotly", "Matplotlib"],
            index=["Plotly", "Matplotlib"].index(cfg["plot_backend"]),
        )

        # 2. Bins
        st.session_state.bins = st.number_input(
            "Points",
            min_value=1,
            max_value=500,
            value=cfg["bins"],
            step=1,
        )

        # 3. Sigma factor
        st.session_state.kernel_factor = st.number_input(
            "Kernel bandwidth",
            min_value=0.0,
            value=cfg["kernel_factor"],
            step=0.1,
        )

        # 3. Sigma factor
        st.session_state.data_percentage = st.number_input(
            "Data Percentage",
            min_value=0.0,
            value=cfg["data_percentage"],
            step=0.1,
        )

        dataset_key = st.selectbox(
            "Dataset",
            options=list(DATASET_OPTIONS.keys()),
            key="new_tab_dataset",
        )

        model_key = st.selectbox(
            "Model",
            options=list(MODEL_OPTIONS.keys()),
            key="new_tab_model",
        )

        # Dynamically update feature list when dataset changes
        if dataset_key:
            dataset_X, _ = DATASET_OPTIONS[dataset_key]
            feature_names = get_feature_names(dataset_X)
        else:
            feature_names = []

        # Use dataset-dependent key so selection resets when dataset changes
        feature_name = st.selectbox(
            "Feature",
            options=feature_names,
            key=f"new_tab_feature_{dataset_key}",
        )

        # ---- Auto-default name: "<dataset> <next_tab_id>" ----
        auto_default_name = f"{dataset_key} {model_key} {feature_name} ({st.session_state.next_tab_id})"

        # Initialize or update the default name when dataset or next_tab_id changes
        if "new_tab_name" not in st.session_state:
            st.session_state.new_tab_name = auto_default_name
        if "new_tab_last_dataset" not in st.session_state:
            st.session_state.new_tab_last_dataset = dataset_key
        if "new_tab_last_id" not in st.session_state:
            st.session_state.new_tab_last_id = st.session_state.next_tab_id
        if "new_tab_last_feature" not in st.session_state:
            st.session_state.new_tab_last_feature = feature_name
        if "new_tab_last_model" not in st.session_state:
            st.session_state.new_tab_last_model = model_key

        # If dataset changed OR id changed (e.g. after creating a tab), refresh default
        if (
            dataset_key != st.session_state.new_tab_last_dataset
            or st.session_state.next_tab_id != st.session_state.new_tab_last_id
            or model_key != st.session_state.new_tab_last_model
            or feature_name != st.session_state.new_tab_last_feature
        ):
            st.session_state.new_tab_name = auto_default_name

        st.session_state.new_tab_last_dataset = dataset_key
        st.session_state.new_tab_last_id = st.session_state.next_tab_id

        tab_name = st.text_input("Tab name", key="new_tab_name")

        create_btn = st.button("Create tab")

        if create_btn:
            if not tab_name:
                st.warning("Please provide a tab name.")
            elif not feature_name:
                st.warning("Please select a feature.")
            else:
                add_new_tab(tab_name, dataset_key, model_key, feature_name)
                st.rerun()


def render_tab_content(tab_id: str, tab_state: TabState) -> None:
    """Render the content inside a single tab."""

    # Editable tab name
    new_name = st.text_input(
        "Tab name",
        value=tab_state.name,
        key=f"tab_name_input_{tab_id}",
    )
    if new_name.strip() and new_name != tab_state.name:
        tab_state.name = new_name.strip()

    # Top-level controls for this tab
    controls_col1, controls_col2, controls_col3 = st.columns([1, 1, 4])
    with controls_col1:
        if st.button("Duplicate (shared core)", key=f"dup_{tab_id}"):
            duplicate_tab(tab_id)
            st.rerun()
    with controls_col2:
        if st.button("Delete", key=f"del_{tab_id}"):
            delete_tab(tab_id)
            st.rerun()

    # Main layout: left column for explainers, central panel for plot
    left_col, center_col = st.columns([1, 3])

    with left_col:
        st.subheader("Explainers")

        updated_explainers: List[str] = []
        for explainer in EXPLAINERS:
            checked = explainer in tab_state.active_explainers
            is_checked = st.checkbox(
                explainer,
                value=checked,
                key=f"{tab_id}_expl_{explainer}",
            )
            if is_checked:
                updated_explainers.append(explainer)

        tab_state.active_explainers = updated_explainers

    with center_col:
        st.subheader("Visualization")

        cfg = get_app_config()

        if not tab_state.active_explainers:
            st.info("Select at least one explainer on the left to generate the figure.")
            return

        try:

            if cfg["plot_backend"] == "Plotly":
                # Plotly-based visualization
                fig = tab_state.core.visualize_doubleplot_plotly(
                    tab_state.active_explainers
                )
                st.plotly_chart(fig, use_container_width=True)

            else:
                fig = tab_state.core.visualize_doubleplot_matplotlib(
                    tab_state.active_explainers
                )
                st.pyplot(fig)

        except Exception as exc:  # pragma: no cover - defensive
            st.error(f"Error while generating visualization: {exc}")


# -----------------------------------------------------------------------------
# MAIN APP ENTRY POINT
# -----------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="ExplanationCore multi-tab app", layout="wide")

    st.title("Explanation dashboard")
    st.caption(
        "Multiple tabs, each with its own ExplanationCore instance."
        " Duplicate tabs share the same core object."
    )

    init_session_state()
    render_sidebar()

    if not st.session_state.tabs:
        st.info(
            "No tabs yet. Use the sidebar to create a new tab by selecting a "
            "dataset, model, and feature."
        )
        return

    # Render all tabs
    tab_ids = list(st.session_state.tabs.keys())
    tab_labels = [st.session_state.tabs[tid].name for tid in tab_ids]

    st_tabs = st.tabs(tab_labels)

    for tid, tab_label, tab_container in zip(tab_ids, tab_labels, st_tabs):
        tab_state = st.session_state.tabs[tid]
        with tab_container:
            render_tab_content(tid, tab_state)


if __name__ == "__main__":
    main()
