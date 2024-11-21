from config import config
import streamlit as st

# Use config for Streamlit settings
st.set_page_config(**config.STREAMLIT_CONFIG)