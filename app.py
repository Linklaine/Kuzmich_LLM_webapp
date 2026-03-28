import streamlit as st

# Define the pages
main_page = st.Page("main_page.py", title="Главная")
about_us = st.Page("about_us.py", title="О нас")
historydl=st.Page("historydl.py", title="История")
settings=st.Page("settings.py", title="Настройки")
prompts=st.Page("prompts.py", title="Промпты")

# Set up navigation
pg = st.navigation([main_page, about_us,historydl,settings,prompts],position="top")

# Run the selected page
pg.run()