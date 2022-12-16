import os
from glob import glob
import json

import numpy as np
import streamlit as st
from pycocotools.coco import COCO



def get_data_folders(dir_path: str) -> list:
    """root 하위 폴더 리스트 반환

    Args:
        dir_path (str): root folder name
    """
    img_paths = os.listdir(dir_path)  # 터미널 실행 위치 기준으로 폴더 상대경로를 지정해야 합니다
    return img_paths


def show_img(img: np.ndarray) -> None:
    """이미지 streamlit에 출력

    Args:
        img (np.ndarray): numpy image
    """
    st.image(img)


def get_img_paths(coco_path: str) -> list[str]:
    """img path 반환

    Args:
        coco_path (str): coco json 파일 경로

    Returns:
        list[str]: 이미지 경로 리스트로 반환
    """
    json_path = os.path.join(glob(f'{coco_path}/*.json')[0])
    data = COCO(json_path)
    img_paths = []
    for img_info in data.loadImgs(data.getImgIds()):
        img_paths.append(os.path.join(coco_path, 'images', img_info['file_name']))

    return img_paths