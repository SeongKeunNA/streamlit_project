import csv
import os
import pickle
import sys
from glob import glob

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from pycocotools.coco import COCO
import cv2
import sys
import pickle
import csv
maxInt = sys.maxsize

from typing import List



while True:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.

    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

CLASS_NAMES = [
    "General trash",
    "Paper",
    "Paper pack",
    "Metal",
    "Glass",
    "Plastic",
    "Styrofoam",
    "Plastic bag",
    "Battery",
    "Clothing",
]

CLASS_COLOR = [
    [1.0, 222 / 255, 1.0],
    [222 / 255, 222 / 255, 1.0],
    [1.0, 1.0, 222 / 255],
    [222 / 255, 222 / 255, 239 / 255],
    [222 / 255, 1.0, 1.0],
    [1.0, 222 / 255, 222 / 255],
    [239 / 255, 222 / 255, 222 / 255],
    [239 / 255, 222 / 255, 1.0],
    [222 / 255, 239 / 255, 239 / 255],
    [222 / 255, 1.0, 222 / 255],
]


def get_listdir(dir_path: str) -> List[str]:
    """root 하위 파일, 폴더 리스트 반환

    Args:
        dir_path (str): root folder name
    """
    dir_names = []
    for dir_name in os.listdir(dir_path):  # 터미널 실행 위치 기준으로 폴더 상대경로를 지정해야 합니다
        if not dir_name.startswith("."):  # hidden file들은 제외합니다
            dir_names.append(dir_name)
    return dir_names


def get_submission_csv(dir_path: str) -> List[str]:
    dir_names = []
    for dir_name in os.listdir(dir_path):
        if not dir_name.startswith(".") and dir_name.endswith(".csv"):
            dir_names.append(dir_name)
            dir_names.sort()
    return dir_names


def show_img(img: np.ndarray) -> None:
    """이미지 streamlit에 출력

    Args:
        img (np.ndarray): numpy image
    """
    st.image(img)


def get_current_page_list(
    img_paths: List[str], page: int = 0, ele_per_page: int = 10
) -> List[str]:
    """현재 페이지에 해당하는 img path들을 반환

    Args:
        img_paths (list): 모든 img path가 있는 list
        page (int, optional): 현재 페이지. Defaults to 0.
        ele_per_page (int, optional): 페이지당 보여줄 이미지 경로의 갯수. Defaults to 10.

    Returns:
        list[str]: 페이지에 해당하는 이미지 경로가 들어있는 리스트
    """
    first_idx = page * ele_per_page
    last_idx = (page + 1) * ele_per_page
    page_list = img_paths[first_idx:last_idx]
    return page_list


def set_session():
    """data app에서 세션 설정"""
    if "page" not in st.session_state:
        st.session_state.page = 0
    if "mode" not in st.session_state:
        st.session_state.mode = "train"
    if "filtering" not in st.session_state:
        st.session_state.filtering = False


def get_mode():
    return st.session_state.mode


def change_mode(mode: str):
    st.session_state.mode = mode


def move_page(page: int):
    """원하는 페이지로 이동

    Args:
        page (int): 이동을 원하는 페이지 번호

    Raises:
        ValueError: page error있는 경우 발생
    """
    try:
        st.session_state.page = page
    except:
        raise ValueError("page에 int가 아닌 값이 들어왔거나 page변수가 존재하지 않습니다.")


@st.experimental_memo
def get_img_paths(coco_path: str, mode: str) -> List[str]:
    """img, id 를 튜플로 담은 리스트를 반환

    Args:
        coco_path (str): json path가 있는 root 폴더 경로
        mode (str): train, val, test

    Returns:
        list: (id, img)를 튜플로 담은 리스트
    """

    data = _get_coco_data(coco_path, mode)
    img_ids_paths = []
    for img_info in data.loadImgs(data.getImgIds()):
        img_ids_paths.append(
            (img_info["id"], os.path.join(coco_path, img_info["file_name"]))
        )
    return img_ids_paths


@st.experimental_singleton
def _get_coco_data(coco_path: str, mode: str) -> COCO:
    """json path로 COCO 클래스 생성

    Args:
        coco_path (str): coco json 경로
        mode (str): train, val, test

    Returns:
        COCO: COCO클래스
    """

    json_path = os.path.join(coco_path, f"{mode}.json")
    data = COCO(json_path)
    return data


@st.experimental_singleton
def get_labeld_img(img, img_id: int, coco_path: str, mode: str) -> np.ndarray:
    label = np.zeros(img.shape)
    data = _get_coco_data(coco_path, mode)
    annos = data.getAnnIds(imgIds=img_id)
    segmentations = sorted(data.loadAnns(annos), key=lambda x: -x["area"])
    for idx in range(len(segmentations)):
        label[data.annToMask(segmentations[idx]) == 1] = CLASS_COLOR[
            segmentations[idx]["category_id"]
        ]
    return label


@st.experimental_singleton
def get_overlay_img(img, img_id: int, coco_path: str, mode: str) -> np.ndarray:
    label = np.zeros_like(img)
    data = _get_coco_data(coco_path, mode)
    annos = data.getAnnIds(imgIds=img_id)
    segmentations = sorted(data.loadAnns(annos), key=lambda x: -x["area"])
    for idx in range(len(segmentations)):
        label[data.annToMask(segmentations[idx]) == 1] = np.array(
            np.array(CLASS_COLOR[segmentations[idx]["category_id"]]) * 255,
            dtype=np.uint8,
        )
    label = cv2.addWeighted(img, 0.5, label, 0.5, 0)
    return label


def get_color_map():
    """read segmentation masking color data

    Returns:
        df:클래스별 r,g,b 값을 갖고있는 데이터 프레임 반환
    """
    category_names = (
        "Background",
        "General trash",
        "Paper",
        "Paper pack",
        "Metal",
        "Glass",
        "Plastic",
        "Styrofoam",
        "Plastic bag",
        "Battery",
        "Clothing",
    )
    r = (0, 166, 31, 178, 51, 251, 227, 253, 255, 202, 106)
    g = (0, 206, 120, 223, 160, 154, 26, 191, 127, 178, 61)
    b = (0, 227, 180, 138, 44, 153, 28, 111, 0, 214, 154)

    df = pd.DataFrame()
    df["name"] = category_names
    df["r"] = r
    df["g"] = g
    df["b"] = b

    return df


def create_trash_label_colormap():
    """Creates a label colormap used in Trash segmentation.

    Returns:
        A colormap for visualizing segmentation results.
    """
    class_colormap = get_color_map()
    colormap = np.zeros((11, 3), dtype=np.uint8)
    for index, (_, r, g, b) in enumerate(class_colormap.values):
        colormap[index] = [r, g, b]

    return colormap


def label_to_color_image(label: np.array):
    """Adds color defined by the dataset colormap to the label.

    Args:
        label: A 2D array with integer type, storing the segmentation label.

    Returns:
        result: A 2D array with floating type. The element of the array
                is the color indexed by the corresponding element in the input label
                to the trash color map.

    Raises:
        ValueError: If label is not of rank 2 or its value is larger than color
              map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError("Expect 2-D input label")

    colormap = create_trash_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError("label value too large.")

    return colormap[label]


# def set_class_checkbox(check: list[bool]):


def get_submission_img(mask: np.ndarray, check: List[bool]) -> np.ndarray:
    """check에서 선택된 category만 mask image로 변환하여 리턴
def get_submission_img(mask: np.ndarray, check: list) -> np.ndarray:
    '''check에서 선택된 category만 mask image로 변환하여 리턴
    Args:
        mask (np.ndarray): mask data
        check (list): mask image로 변환할 category
    Return:
        mask_image (np.ndarray): 선택된 category만 표현한 mask image
    """
    for idx, i in enumerate(check):
        if i == False:
            mask[mask == idx] = 0
    return label_to_color_image(mask)


@st.experimental_singleton
def get_coco_img(coco_path: str, mode: str, id: int, check: List[bool]):
    """make mask image

    Args:
        coco_path (str): coco json 경로
        mode (str): train, val, test
        id (int): image_id in annotation
        check (List[bool]): 카테고리 별로 checkbox가 check됬는지를 담은 리스트

    Returns:
        mask (np.array): check된 category에 대한 mask 이미지
    """
    data = _get_coco_data(coco_path, mode)

    ann_ids = data.getAnnIds(id)
    anns = data.loadAnns(ann_ids)

    mask = np.zeros((512, 512))

    for i in range(len(anns)):
        class_idx = anns[i]["category_id"]
        if check[class_idx] == False:
            continue
        pixel_value = class_idx
        mask[data.annToMask(anns[i]) == 1] = pixel_value
    mask = mask.astype(np.int8)
    mask = label_to_color_image(mask)
    return mask


def get_submission_category(mask: np.ndarray) -> List[bool]:
    """submission mask data에 포함된 category list
    Args:
        mask (np.ndarray): mask data
        category (list): mask data에 포함된 category list
    Return:
        output_dict (dict): test image별 mask data를 포함한 dict
    """
    cat_available = np.unique(mask)
    category = [0] * 11
    for i in cat_available:
        category[i] = 1
    category[0] = 0
    return category


@st.experimental_singleton
def get_coco_category(coco_path: str, mode: str, id: int) -> List[bool]:
    """선택된 이미지에(베이스 이미지) 대하여 활성화 되어있는 category를 구함

    Args:
        coco_path (str): coco json 경로
        mode (str): train, val, test
        id (int): image_id in annotation

    Returns:
        category (List[int]): 각각의 카테고리에 대하여 활성화 되어있는지 여부를 담은 리스트

    """
    data = _get_coco_data(coco_path, mode)
    ann_ids = data.getAnnIds(id)
    anns = data.loadAnns(ann_ids)

    category = [0] * 11
    for i in range(len(anns)):
        category[anns[i]["category_id"]] = 1
    return category


def overlay_image(base: np.array, mask: np.array):
    """base이미지와 mask이미지를 이용하여 overlay 이미지 생성

    Args:
        base (np.array): base image
        mask (np.array): mask image

    Returns:
        img (np.array): overlay image

    """
    img = cv2.addWeighted(base, 0.5, mask, 0.5, 0)
    return img


def erase_image(coco_path: str, img_id: int):
    """modify 파일에 지울 이미지 번호 저장

    Args:
        coco_path (str): coco json 경로
        img_id (int): 이미지 번호
    """
    with open(os.path.join(coco_path, "modify.json"), "a") as f:
        f.write(str(img_id) + "\n")


def make_checkbox(valid_category: list):
    """각 카테고리에 대한 checkbox 생성
    Args:
        valid_category (list[int]): class_id list
    Return:
        return_list (List[int]): 각 클래스별 checkbox 체크 여부 list
    """
    classes = (
        "Background",
        "General trash",
        "Paper",
        "Paper pack",
        "Metal",
        "Glass",
        "Plastic",
        "Styrofoam",
        "Plastic bag",
        "Battery",
        "Clothing",
    )
    check_boxes = st.columns(4)
    return_list = [False] * len(classes)
    for idx, class_name in enumerate(classes):
        with check_boxes[idx % 4]:
            if valid_category[idx] == 1:
                check = st.checkbox(class_name, value=True)
            else:
                check = st.checkbox(class_name, value=False, disabled=True)
        return_list[idx] = check
    return return_list


def load_pkl(path: str) -> any:
    with open(path, "rb") as f:
        data = pickle.load(f)
        return data


def save_pkl(path: str, data: any) -> None:
    cache_file_path = path.split(".")[0] + ".pkl"
    with open(cache_file_path, "wb") as f:
        pickle.dump(data, f)


def submission_to_dict(submission_path: str) -> dict:
    """submission.csv를 dict로 변환
    Args:
        submission_path (str): submission.csv file path
    Return:
        output_dict (dict): test image별 mask data를 포함한 dict
    """
    output_dict = {}
    size = 256
    with open(submission_path, mode="r") as file:
        reader = csv.reader(file)
        attrs = []
        for row in reader:
            if len(attrs) < 1:
                attrs = row
                continue
            filename = row[0]
            mask = np.array(list(map(int, row[1].split()))).reshape([size, size])
            output_dict[filename] = mask
    return output_dict


def load_submission_dict(submission_path: str) -> dict:
    """submission.csv에 포함된 mask data를 dict로 load, 최초 실행 시 dict를 pkl로 변환, 이후는 pkl load
    Args:
        submission_path (str): submission.csv file path
    Return:
        output_dict (dict): test image별 mask data를 포함한 dict
    """
    cache_file_path = submission_path.split(".")[0] + ".pkl"
    if os.path.isfile(cache_file_path):
        return load_pkl(cache_file_path)
    else:
        output_dict = submission_to_dict(submission_path)
        save_pkl(cache_file_path, output_dict)
        return load_submission_dict(submission_path)


@st.experimental_memo
def class_filtering(coco_path: str, mode: str, img_ids_paths: list, showed_cls: list) -> List[tuple]:
    filtered_list = []
    data = _get_coco_data(coco_path, mode)
    for (id, path) in img_ids_paths:
        annos = data.getAnnIds(id)
        infos = data.loadAnns(annos)
        for info in infos:
            if CLASS_NAMES[info["category_id"] - 1] in showed_cls:
                filtered_list.append((id, path))
                break
    return filtered_list


def filter_mode_change():
    st.session_state.filtering = not st.session_state.filtering

def get_data_folders(dir_path: str) -> list:
    """root 하위 폴더 리스트 반환
    Args:
        dir_path (str): root folder name
    """
    dir_names = []
    for dir_name in os.listdir(dir_path):   # 터미널 실행 위치 기준으로 폴더 상대경로를 지정해야 합니다
        if not dir_name.startswith('.'):    # hidden file들은 제외합니다
            dir_names.append(dir_name)
    return dir_names