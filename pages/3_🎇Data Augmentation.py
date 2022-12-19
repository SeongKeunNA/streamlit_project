import streamlit as st
import os
import cv2
from math import ceil
from time import time

from utils.data import *
from utils.aug import *

ROOTDIR = "data/trash"
ELEMENTS_PER_PAGE = 10
WIDTH_ORIGINAL = 400

st.set_page_config(page_title="Data Augmentation", page_icon="🎇", layout="wide")
set_session()

st.title("Data Augmentation")

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

placeholder_params = get_placeholder_params(img)

augmentations = load_augmentations_config(
                placeholder_params, "configs/augmentations.json"
            )


# get the list of transformations names
transform_names = select_transformations(augmentations)

# get parameters for each transform
transforms = get_transormations_params(transform_names, augmentations)

try:
    # apply the transformation to the image
    data = A.ReplayCompose(transforms)(image=img)
    error = 0
except ValueError:
    error = 1
    st.title(
        "The error has occurred. Most probably you have passed wrong set of parameters. \
    Check transforms that change the shape of image."
    )

# proceed only if everything is ok
if error == 0:
    augmented_image = data["image"]

    # show the images
    width_transformed = int(
        WIDTH_ORIGINAL / img.shape[1] * augmented_image.shape[1]
    )

    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Original image", width=WIDTH_ORIGINAL)
        st.write(img.shape)
    with col2:
        st.image(
            augmented_image,
            caption="Transformed image",
            width=width_transformed,
        )
        st.write(augmented_image.shape)
    end_time = time()

    st.write(f"loading time: {end_time - start_time:.3f}")
    # comment about refreshing
    st.write("*Press 'R' to refresh*")

    # random values used to get transformations
    show_random_params(data)

    # print additional info
    for transform in transforms:
        show_docstring(transform)
        st.code(str(transform))
    show_credentials()

    # adding google analytics pixel
    # only when deployed online. don't collect statistics of local usage
    if "GA" in os.environ:
        st.image(os.environ["GA"])
        st.markdown(
            (
                "[Privacy policy]"
                + (
                    "(https://htmlpreview.github.io/?"
                    + "https://github.com/IliaLarchenko/"
                    + "albumentations-demo/blob/deploy/docs/privacy.html)"
                )
            )
        )
