import streamlit as st
import os
import cv2
from math import ceil
from time import time
from utils.data import *
from utils.distribution import *

ROOTDIR = "data"
# TODO 
# 1. 클래스 개수의 분포
# 2. 이미지 당 클래스 개수의 분포
# 3. 클래스 별 전체 영역 대비 픽셀 비율 분포
# 4. 클래스 픽셀의 컬러 정보 분포

st.set_page_config(page_title="Data Distrubution") #,page_icon="📊")
set_session()

st.title("Data Distribution")

with st.sidebar:
    selected_dir = st.selectbox(
        label="select folder", options=get_data_folders(ROOTDIR)
    )
        
    selected_category = st.selectbox(
        label="select category", options=["Proportion distribution"]
    )
    
start_time = time()
show_distribution(selected_dir, selected_category)
end_time = time()

st.write(f"loading time: {end_time - start_time:.3f}")