import streamlit as st
import os
import cv2
from math import ceil
from time import time

from utils.data import *
from utils.aug import *

ROOTDIR = "data/trash"
SUBMISSIONS_ROOT_DIR = os.path.join(ROOTDIR, "submissions")
ELEMENTS_PER_PAGE = 10

st.set_page_config(page_title="Segmentation Viewer", page_icon="✂️")
set_session()

st.title("Segmentation Viewer")

with st.sidebar:
    selected_mode = st.selectbox(
        label="select dataset", options=["train", "val", "test"]
    )

    if selected_mode == "test":
        submission_filename = st.selectbox(label="select_submission", options=get_listdir(SUBMISSIONS_ROOT_DIR))
        
    img_ids_paths = get_img_paths(ROOTDIR, selected_mode)

    img_id_path = st.radio(
        label="사진 선택",
        options=get_current_page_list(
            img_ids_paths, st.session_state.page, ELEMENTS_PER_PAGE
        ),
        format_func=lambda x: f"{x[1].split('/')[-1]}",
    )
    img_id, img_path = img_id_path
    page2move = st.slider(
        label="페이지를 선택하세요",
        min_value=1,
        max_value=ceil(len(img_ids_paths) / ELEMENTS_PER_PAGE),
    )
    st.button(label="이동", on_click=move_page, args=([page2move - 1]))

start_time = time()

valid_category = get_coco_category(ROOTDIR, selected_mode, img_id)
check = make_checkbox(valid_category)
base_img = cv2.imread(img_path)
base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)

if selected_mode == "test":
    base_img = cv2.resize(base_img, (256,256))
    submission_path = os.path.join(SUBMISSIONS_ROOT_DIR, submission_filename)
    mask_dict = load_submission_dict(submission_path)
    submission_index = "/".join(img_path.split("/")[2:])
    mask_img = label_to_color_image(mask_dict[submission_index])
else:
    mask_img = get_coco_img(ROOTDIR, selected_mode, img_id, check)

overlay_img = overlay_image(base_img, mask_img)

col1, col2, col3 = st.columns(3)

with col1:
    show_img(base_img)

with col2:
    show_img(mask_img)

with col3:
    show_img(overlay_img)


end_time = time()

click = st.button("erase")
if click:
    erase_image(ROOTDIR, img_id)


st.write(f"loading time: {end_time - start_time:.3f}")
