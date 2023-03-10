import os
from math import ceil
from time import time

import cv2
import streamlit as st
from utils.aug import *
from utils.data import *
import re

ROOTDIR = "data/trash"
SUBMISSIONS_ROOT_DIR = os.path.join(ROOTDIR, "submissions")
ELEMENTS_PER_PAGE = 10

st.set_page_config(page_title="Data Viewer", page_icon="๐")
set_session()

st.title("Segmentation Viewer")

with st.sidebar:
    selected_mode = st.selectbox(
        label="select dataset", options=["train", "val", "test"]
    )

    if get_mode() != selected_mode:
        change_mode(selected_mode)
        st.session_state.page = 0

    if selected_mode == "test":
        submission_filename = st.selectbox(
            label="select_submission", options=get_submission_csv(SUBMISSIONS_ROOT_DIR)
        )

    img_ids_paths = get_img_paths(ROOTDIR, selected_mode)
    st.info("ํ๋๋ผ๋ ์ ํ๋ ์นดํ๊ณ ๋ฆฌ๊ฐ ๋ค์ด๊ฐ์์ผ๋ฉด ๊ทธ ์ด๋ฏธ์ง๊ฐ ๋ฆฌ์คํธ์ ๋ค์ด๊ฐ๋๋ค", icon="โน๏ธ")
    showed_cls = st.multiselect(
        label="์์",
        options=CLASS_NAMES,
        label_visibility="collapsed",
        default=CLASS_NAMES,
    )

    if st.session_state.filtering:
        button_text = "๋ฏธ์ ์ฉ"
    else:
        button_text = "์ ์ฉ"
    st.button(label=button_text, on_click=filter_mode_change)

    if st.session_state.filtering:
        img_ids_paths = class_filtering(
            ROOTDIR, selected_mode, img_ids_paths, showed_cls
        )

    img_id_path = st.radio(
        label="์ฌ์ง ์ ํ",
        options=get_current_page_list(
            img_ids_paths, st.session_state.page, ELEMENTS_PER_PAGE
        ),
        format_func=lambda x: f"{x[1].split('/')[-2:]}",
    )

    img_id, img_path = img_id_path

    page2move = st.slider(
        label="ํ์ด์ง๋ฅผ ์ ํํ์ธ์",
        min_value=1,
        max_value=ceil(len(img_ids_paths) / ELEMENTS_PER_PAGE),
    )

    st.button(label="์ด๋", on_click=move_page, args=([page2move - 1]))

start_time = time()

base_img = cv2.imread(img_path)
base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)

if selected_mode == "test":
    base_img = cv2.resize(base_img, (256, 256))
    submission_path = os.path.join(SUBMISSIONS_ROOT_DIR, submission_filename)
    mask_dict = load_submission_dict(submission_path)
    submission_index = "/".join(re.split('/|\\\\', img_path)[2:])
    valid_category = get_submission_category(mask_dict[submission_index])
    check = make_checkbox(valid_category)
    mask_img = get_submission_img(mask_dict[submission_index], check)
else:
    valid_category = get_coco_category(ROOTDIR, selected_mode, img_id)
    check = make_checkbox(valid_category)
    mask_img = get_coco_img(ROOTDIR, selected_mode, img_id, check)

overlay_img = overlay_image(base_img, mask_img)

col1, col2, col3 = st.columns(3)

with col1:
    show_img(base_img)

with col2:
    show_img(mask_img)

with col3:
    show_img(overlay_img)

from annotated_text import annotation

label_colors = [0] * 10
for idx, v in enumerate(check[1:]):
    if v:
        label_colors[idx] = 1

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    if label_colors[0]:
        st.markdown(
            annotation("General trash", background="#a6cee3", color="black"),
            unsafe_allow_html=True,
        )
    if label_colors[1]:
        st.markdown(
            annotation("Paper", background="#1f78b4", color="black"),
            unsafe_allow_html=True,
        )

with col2:
    if label_colors[2]:
        st.markdown(
            annotation("Paper pack", background="#b2df8a", color="black"),
            unsafe_allow_html=True,
        )
    if label_colors[3]:
        st.markdown(
            annotation("Metal", background="#33a02c", color="black"),
            unsafe_allow_html=True,
        )

with col3:
    if label_colors[4]:
        st.markdown(
            annotation("Glass", background="#fb9a99", color="black"),
            unsafe_allow_html=True,
        )
    if label_colors[5]:
        st.markdown(
            annotation("Plastic", background="#e31a1c", color="black"),
            unsafe_allow_html=True,
        )
with col4:
    if label_colors[6]:
        st.markdown(
            annotation("Styrofoam", background="#fdbf6f", color="black"),
            unsafe_allow_html=True,
        )
    if label_colors[7]:
        st.markdown(
            annotation("Plastic bag", background="#ff7f00", color="black"),
            unsafe_allow_html=True,
        )
with col5:
    if label_colors[8]:
        st.markdown(
            annotation("Battery", background="#cab2d6", color="black"),
            unsafe_allow_html=True,
        )
    if label_colors[9]:
        st.markdown(
            annotation("Clothing", background="#6a3d9a", color="black"),
            unsafe_allow_html=True,
        )


end_time = time()

click = st.button("erase")
if click:
    erase_image(ROOTDIR, img_id)


st.write(f"loading time: {end_time - start_time:.3f}")
