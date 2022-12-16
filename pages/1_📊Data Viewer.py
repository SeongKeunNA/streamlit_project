import streamlit as st
import os

from utils.data import get_data_folders, get_img_paths

ROOTDIR = "data"

st.set_page_config(page_title="Data Viewer", page_icon="📊")

st.title("Data Viewer")

with st.sidebar:
    selected_dir = st.selectbox(
        label="select folder", options=get_data_folders(ROOTDIR)
    )
    img_paths = get_img_paths(os.path.join(ROOTDIR, selected_dir))
    st.radio(label="사진 선택",options=img_paths[:10])
