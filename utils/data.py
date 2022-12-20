import os
from glob import glob
import pandas as pd
import numpy as np
import streamlit as st
from pycocotools.coco import COCO
import cv2

CLASS_COLOR = [
    [1., 222/255, 1.],
    [222/255, 222/255, 1.],
    [1., 1., 222/255],
    [222/255, 222/255, 239/255],
    [222/255, 1., 1.],
    [1., 222/255, 222/255],
    [239/255, 222/255, 222/255],
    [239/255, 222/255, 1.],
    [222/255, 239/255, 239/255],
    [222/255, 1., 222/255],
]

def get_data_folders(dir_path: str) -> list:
    """root 하위 폴더 리스트 반환

    Args:
        dir_path (str): root folder name
    """
    dir_names = []
    for dir_name in os.listdir(dir_path):  # 터미널 실행 위치 기준으로 폴더 상대경로를 지정해야 합니다
        if not dir_name.startswith("."):  # hidden file들은 제외합니다
            dir_names.append(dir_name)
    return dir_names


def show_img(img: np.ndarray) -> None:
    """이미지 streamlit에 출력

    Args:
        img (np.ndarray): numpy image
    """
    st.image(img)


def get_current_page_list(
    img_paths: list, page: int = 0, ele_per_page: int = 10
) -> list[str]:
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
        st.session_state.mode = 'train'


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
def get_img_paths(coco_path: str, mode: str) -> list:
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
    segmentations = sorted(data.loadAnns(annos), key=lambda x: -x['area'])
    for idx in range(len(segmentations)):
        label[data.annToMask(segmentations[idx]) == 1] = CLASS_COLOR[segmentations[idx]['category_id']]
    return label


@st.experimental_singleton
def get_overlay_img(img, img_id: int, coco_path: str, mode: str) -> np.ndarray:
    label = np.zeros_like(img)
    data = _get_coco_data(coco_path, mode)
    annos = data.getAnnIds(imgIds=img_id)
    segmentations = sorted(data.loadAnns(annos), key=lambda x: -x['area'])
    for idx in range(len(segmentations)):
        label[data.annToMask(segmentations[idx]) == 1] = np.array(np.array(CLASS_COLOR[segmentations[idx]['category_id']])*255, dtype=np.uint8)
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
    r = (0, 192, 0, 0, 128, 64, 64, 192, 192, 64, 128)
    g = (0, 0, 128, 128, 0, 0, 0, 192, 192, 64, 0)
    b = (0, 128, 192, 64, 0, 128, 192, 64, 128, 128, 192)

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


@st.experimental_singleton
def get_coco_img(coco_path: str, mode: str, id: int, check: list[bool]):
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


@st.experimental_singleton
def get_coco_category(coco_path: str, mode: str, id: int):
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


def make_checkbox(valid_category: list[int]):
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
