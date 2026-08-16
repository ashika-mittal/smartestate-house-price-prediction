from importlib import reload

import streamlit as st

from src import ui


# Streamlit may rerun this entrypoint while retaining imported local modules.
# Reloading this small UI-only module keeps live design edits in sync.
ui = reload(ui)


st.set_page_config(
    page_title="SmartEstate",
    page_icon=":material/home_work:",
    layout="wide",
)

st.session_state.setdefault("dark_mode", False)
st.session_state.setdefault("prediction_result", None)

predict_page = st.Page(
    "app_pages/predict.py",
    title="Predict",
    icon=":material/calculate:",
    default=True,
)
insights_page = st.Page(
    "app_pages/insights.py",
    title="Insights & assistant",
    icon=":material/auto_awesome:",
)

page = st.navigation(
    [predict_page, insights_page],
    position="top",
)

ui.apply_ui_theme(st.session_state.dark_mode)

with st.container(key="brand_shell"):
    brand_column, theme_column = st.columns(
        [4.2, 1.35],
        gap="large",
        vertical_alignment="center",
    )

    with brand_column:
        ui.render_brand_header()

    with theme_column:
        with st.container(border=True, key="theme_control", gap="xxsmall"):
            st.toggle(
                ":material/dark_mode: Dark mode",
                key="dark_mode",
                help="Switch between bright and dark appearance.",
                width="stretch",
                persist_state="session",
            )

page.run()
ui.render_footer()
