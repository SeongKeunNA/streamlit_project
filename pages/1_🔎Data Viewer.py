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
    
    selected_mode = st.selectbox(
        label="select dataset", options=['train', 'val', 'test']
    )
    start_time = time()
    img_ids_paths = get_img_paths(ROOTDIR, selected_mode)
    end_time = time()
    st.text(f'coco data 로딩 시간: {end_time-start_time:.3f}')
    img_id_path = st.radio(
        label="사진 선택",
        options=get_current_page_list(img_ids_paths, st.session_state.page, ELEMENTS_PER_PAGE),
        format_func=lambda x: f"{x[1].split('/')[-1]}")
    
    img_id, img_path = img_id_path
    
    page2move = st.slider(label='페이지를 선택하세요', min_value=1, max_value=ceil(len(img_ids_paths)/ELEMENTS_PER_PAGE))
    
    st.button(label='이동', on_click=move_page, args=([page2move - 1]))




start_time = time()
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
if selected_mode != 'test':
    pass
show_img(img)
end_time = time()

st.write(f"loading time: {end_time - start_time:.3f}")
