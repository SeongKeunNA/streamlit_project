import streamlit as st
import os
import cv2
from math import ceil
from time import time
from utils.data import *
from utils.distribution import *

ROOTDIR = "data"

st.set_page_config(page_title="Data Distrubution") #,page_icon="📊")

st.title("Data Distribution")

with st.sidebar:
    selected_dir = "trash"
        
    selected_category = st.selectbox(
        label="select category", options=["Segmentation proportion distribution", "Color distribution", "Class number distribution", "Object number per image distribution"]
    )
    
start_time = time()
show_distribution(selected_dir, selected_category)
end_time = time()

st.write(f"loading time: {end_time - start_time:.3f}")