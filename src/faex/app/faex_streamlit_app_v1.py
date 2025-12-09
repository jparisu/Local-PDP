"""Streamlit app for managing multiple ExplanationCore-based tabs.

Each tab holds an ExplanationCore instance and a set of active explainers.
Tabs can be created, duplicated (sharing the same core object), or deleted.

You must provide:
- DATASET_OPTIONS: mapping from human-readable dataset name to dataset object
- MODEL_OPTIONS: mapping from human-readable model name to a fitted/ready model
- ExplanationCore implementation and its constructor signature

Assumed ExplanationCore signature (adapt as needed):
    ExplanationCore(dataset, model, feature_name, dataset_name=None, model_name=None)

Assumed visualization method:
    core.visualize_doubleplot_matplotlib(explanations: list[str]) -> matplotlib.figure.Figure
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List

import streamlit as st

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score

from faex.core.DataCore import DataCore
from faex.core.ExplanationCore import ExplanationCore
from faex.resources.dataseting import bikes
from faex.explaining.ExplainerFactory import GlobalExplainerFactory


# -----------------------------------------------------------------------------
# PLACEHOLDERS / CONFIG (adapt to your project)
# -----------------------------------------------------------------------------

# Example placeholders. Replace these with your real objects.
# from your_package.explanations import ExplanationCore
# from your_package.data import bikes, ...
# from sklearn.ensemble import RandomForestRegressor

# Example globals to be edited by you
DATASET_OPTIONS = {
    "Bike sharing": bikes,
}

MODEL_OPTIONS = {
    "Random Forest Regressor 20": RandomForestRegressor(n_estimators=20, random_state=42),
    "Random Forest Regressor 100": RandomForestRegressor(n_estimators=100, random_state=42),
    "Neural Network Regressor 5-5": MLPRegressor(max_iter=1000, hidden_layer_sizes=(5, 5)),
    "Neural Network Regressor 10-10": MLPRegressor(max_iter=1000, hidden_layer_sizes=(10, 10)),
}

EXPLAINERS: List[str] = GlobalExplainerFactory().get_available_plot_explainers()


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
    """Infer feature names from a dataset object.

    Adjust this to your project. By default it tries:
    - pandas.DataFrame.columns
    - dict keys
    Otherwise returns a small dummy list.
    """

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
    # 5-fold cross-validation on the full dataset (test R2 estimate)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="r2")

    # Fit final model on the full dataset (training R2)
    model.fit(X, y)
    return model, cv_scores


def create_core(dataset_key: str, model_key: str, feature_name: str) -> Any:
    """Factory for ExplanationCore.

    Edit this to match your actual ExplanationCore constructor.
    """

    df_x, df_y = DATASET_OPTIONS[dataset_key]()
    model_obj = MODEL_OPTIONS[model_key]

    model, score = train_model(model_obj, df_x, df_y)

    # Replace this with your real import & constructor
    # Example:
    # from your_package.explanations import ExplanationCore
    # return ExplanationCore(dataset_obj, model_obj, feature_name,
    #                        dataset_name=dataset_key,
    #                        model_name=model_key)

    # Placeholder implementation so the template runs without your core:
    datacore = DataCore(
        model=model,
        df_X=df_x,
        study_features=[feature_name],
        bins=100,
    )
    return ExplanationCore(datacore)


def add_new_tab(tab_name: str, dataset_key: str, model_key: str, feature_name: str) -> None:
    """Create a new tab with its own ExplanationCore instance."""

    core = create_core(dataset_key, model_key, feature_name)
    tab_id = str(st.session_state.next_tab_id)
    st.session_state.next_tab_id += 1

    st.session_state.tabs[tab_id] = TabState(name=tab_name, core=core)
    st.session_state.active_tab_id = tab_id


def duplicate_tab(tab_id: str) -> None:
    """Duplicate a tab while sharing the same core object.

    The duplicate receives a new TabState that *references* the same core.
    """

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

    # Update active tab if needed
    if st.session_state.active_tab_id == tab_id:
        if st.session_state.tabs:
            # pick any remaining tab as active
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

    with st.sidebar.expander("Create new tab", expanded=True):
        with st.form("new_tab_form", clear_on_submit=True):
            default_name = f"Tab {st.session_state.next_tab_id}"
            tab_name = st.text_input("Tab name", value=default_name)

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

            if dataset_key:
                dataset_X, _ = DATASET_OPTIONS[dataset_key]()
                feature_names = get_feature_names(dataset_X)
            else:
                feature_names = []

            feature_name = st.selectbox(
                "Feature",
                options=feature_names,
                key="new_tab_feature",
            )

            create_btn = st.form_submit_button("Create tab")

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

    st.markdown(f"### {tab_state.name}")

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

        # persist selection back into tab state
        tab_state.active_explainers = updated_explainers

    with center_col:
        st.subheader("Visualization")

        if not tab_state.active_explainers:
            st.info("Select at least one explainer on the left to generate the figure.")
            return

        try:
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
