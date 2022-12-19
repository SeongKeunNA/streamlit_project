import streamlit as st
import os
import cv2
from math import ceil
from time import time

from utils.data import *

ROOTDIR = "data/trash"
ELEMENTS_PER_PAGE = 10

st.set_page_config(page_title="Data Viewer", page_icon="🔎")
set_session()

st.title("Data Viewer")

with st.sidebar:
    
    selected_dir = st.selectbox(
        label="select dataset", options=['train', 'val', 'test']
    )
    
    img_paths = get_img_paths(ROOTDIR, selected_dir)
    
    img_path = st.radio(
        label="사진 선택",
        options=get_current_page_list(img_paths, st.session_state.page, ELEMENTS_PER_PAGE),
        format_func=lambda x: f"{x.split('/')[-1]}")
    
    page2move = st.slider(label='페이지를 선택하세요', min_value=1, max_value=ceil(len(img_paths)/ELEMENTS_PER_PAGE))
    
    st.button(label='이동', on_click=move_page, args=([page2move - 1]))

start_time = time()
print(img_path)
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
show_img(img)
end_time = time()

st.write(f"loading time: {end_time - start_time:.3f}")
