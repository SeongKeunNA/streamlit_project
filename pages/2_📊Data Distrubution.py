import streamlit as st
import os
import cv2
from math import ceil
from time import time

from utils.data import *

ROOTDIR = "data"
# TODO
#
# Class distribution
# Class number distribution per image
# Color distribution by class

st.set_page_config(page_title="Data Distrubution") #,page_icon="📊")
set_session()

st.title("Data Distribution")

with st.sidebar:
    selected_dir = st.selectbox(
        label="select folder", options=get_data_folders(ROOTDIR)
    )
        
    selected_category = st.selectbox(
        label="select category", options=["Class distribution", "Class number distribution per image", "Color distribution by class"]
    )
    
start_time = time()
# show_distribution(selected_dir, selected_category)
end_time = time()

st.write(f"loading time: {end_time - start_time:.3f}")