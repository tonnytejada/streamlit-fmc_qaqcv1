import streamlit as st
# --- PAGE SETUP ---
about_page = st.Page(
    "views/about_me.py",
    title="About Me",
    icon=":material/account_circle:",
    default=True,
)

project_1_page = st.Page(
    "views/qaqc.py",
    title="Duplicate analysis",
    icon=":material/view_kanban:",
)

project_2_page = st.Page(
    "views/blank.py",
    title="Blank analysis",
    icon=":material/flag:",
)

# --- NAVIGATION SETUP [WITHOUT SECTIONS] ---
#pg = st.navigation(pages=[about_page, project_1_page, project_2_page])

# --- NAVIGATION SETUP [WITH SECTIONS]---
pg = st.navigation(
    {
        "Information": [about_page],
        "QA/QC": [project_1_page, project_2_page]
        #"Chat": [project_2_page],
        #"QaQc": [project_3_page],
    }
)

# --- RUN NAVIGATION ---
pg.run()
