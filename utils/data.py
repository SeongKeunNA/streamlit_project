import os
from glob import glob
import json

import numpy as np
import streamlit as st
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
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
    for dir_name in os.listdir(dir_path):   # 터미널 실행 위치 기준으로 폴더 상대경로를 지정해야 합니다
        if not dir_name.startswith('.'):    # hidden file들은 제외합니다
            dir_names.append(dir_name)
    return dir_names


def show_img(img: np.ndarray) -> None:
    """이미지 streamlit에 출력

    Args:
        img (np.ndarray): numpy image
    """
    st.image(img)


def get_current_page_list(img_paths: list, page: int = 0, ele_per_page: int = 10) -> list[str]:
    """현재 페이지에 해당하는 img path들을 반환

    Args:
        img_paths (list): 모든 img path가 있는 list
        page (int, optional): 현재 페이지. Defaults to 0.
        ele_per_page (int, optional): 페이지당 보여줄 이미지 경로의 갯수. Defaults to 10.

    Returns:
        list[str]: 페이지에 해당하는 이미지 경로가 들어있는 리스트
    """
    first_idx = page*ele_per_page
    last_idx = (page+1)*ele_per_page
    page_list = img_paths[first_idx:last_idx]
    return page_list
    

def set_session():
    """data app에서 세션 설정
    """
    if 'page' not in st.session_state:
        st.session_state.page = 0
        
        
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
        raise ValueError('page에 int가 아닌 값이 들어왔거나 page변수가 존재하지 않습니다.')


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
        img_ids_paths.append((img_info['id'], os.path.join(coco_path, img_info['file_name'])))
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
    
    json_path = os.path.join(coco_path, f'{mode}.json')
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

