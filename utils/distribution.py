from pycocotools.coco import COCO
from glob import glob
import pandas as pd
import os
import cv2
import streamlit as st
import json
import numpy as np
from typing import List
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.offline as pyo
import plotly.graph_objs as go
import pickle
import sys
from stqdm import stqdm
from collections import Counter

def set_data(dir:str) -> pd.DataFrame:
    """데이터 설정

    Args:
        dir (str): 데이터셋 경로

    Returns:
        pd.DataFrame: coco format의 annotations들을 하나의 행으로 하는 데이터프레임
    """

    coco = COCO(dir)

    df = pd.DataFrame()

    image_ids = []
    category_ids = []
    category_names = []
    segmentations = []
    areas = []
    sizes = []

    for image_id in coco.getImgIds():

        image_info = coco.loadImgs(image_id)[0]
        ann_ids = coco.getAnnIds(imgIds=image_info["id"])
        anns = coco.loadAnns(ann_ids)

        file_name = image_info["file_name"]
        for ann in anns:
            image_ids.append(file_name)
            category_ids.append(ann["category_id"])
            category_names.append(coco.loadCats(ann["category_id"])[0]["name"])
            segmentations.append(ann['segmentation'])
            areas.append(ann["area"])
            sizes.append(image_info["width"] * image_info["height"])
            

    df["image_id"] = image_ids
    df["category_id"] = category_ids
    df["category_name"] = category_names
    df["segmentation"] = segmentations
    df["area"] = areas
    df["size"] = sizes
    return df
    
def show_distribution(dir:str, category:str):
    """해당 폴더의 해당 카테고리의 distribution image를 반환

    Args:
        dir (str): distribution을 확인하려는 폴더 경로
        category (str): distribution을 확인하려는 카테고리
    """
    
    if f"{dir}_{category}" not in os.listdir("utils/distribution_plotly") or st.button("refresh"):
        make_dist_figs(dir, category)
    display_plotly_figs(f"utils/distribution_plotly/{dir}_{category}")
    
        

def display_plotly_figs(figs_path: str):
    """plotly figures를 streamlit page에 표시하는 함수, plotly figure list가 저장된 pickle 파일을 불러옴
    Args:
        figs_path: pickle 파일 경로
    """
    with open(figs_path, "rb") as fr:
        dist_figs = pickle.load(fr)    
    for dist_fig in dist_figs:
        st.plotly_chart(dist_fig)


def make_dist_figs(dir:str, category:str):
    """해당 폴더의 해당 카테고리의 distribution plot를 계산 및 저장
    Args:
        dir (str): distribution을 확인하려는 폴더 경로
        category (str): distribution을 확인하려는 카테고리
    """
    json_path = os.path.join(glob(f'data/{dir}/*.json')[0])
    df = set_data(json_path)
    if category == "Proportion distribution":
        make_prop_dist_figs(dir, df)
    elif category == "Color distribution":
        make_color_dist_figs(dir, df)
    elif category == "Class distribution":
        make_class_dist_figs(dir, df)
    elif category == "Class number distribution per image":
        make_class_num_dist_figs(dir, df)
    

def make_prop_dist_figs(dir, df):
    areas = dict()
    fig_list = []
    for row in df.itertuples():   
        if row[3] in areas.keys():
            areas[row[3]].append(row[5] / row[6] * 100)
        else:
            areas[row[3]] = [row[5] / row[6] * 100]
    
    for area in areas.keys():
        fig = px.histogram(areas[area], labels = {"value" : "proportion"}, title = area)
        fig.update_layout(showlegend=False)
        fig_list.append(fig)
        
    with open(f"utils/distribution_plotly/{dir}_proportion distribution", "wb") as fw:    
         pickle.dump(fig_list, fw)
        

def make_color_dist_figs(dir, df: pd.DataFrame):
    """bbox 내의 color distribution의 box plot 계산 및 저장
    Args:
        df: coco dataset의 annotations를 각 행으로 하는 데이터 프레임
    """
    color_list = ["r_mean", "g_mean", "b_mean", "h_mean", "s_mean", "v_mean"]
    group = df.groupby("image_id")
    img_paths = list(group.groups.keys())
    len_df = len(df)
    color_ann_cumulation = {color: [0] * len_df for color in color_list}

    for img_path in stqdm(img_paths):
        img_bgr = cv2.imread(os.path.join("../dataset/", img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        bboxes = get_bbox(group.get_group(img_path))
        for bbox in bboxes:
            b_id, c_id, x_min, y_min, x_max, y_max = map(int, bbox)
            cropped_rgb = img_rgb[y_min:y_max, x_min:x_max]
            rgb_mean = np.mean(cropped_rgb, axis=(0, 1))
            cropped_hsv = img_hsv[y_min:y_max, x_min:x_max]
            hsv_mean = np.mean(cropped_hsv, axis=(0, 1))
            color_mean = np.concatenate((rgb_mean, hsv_mean))
            for i, color in enumerate(color_list):
                color_ann_cumulation[color][b_id] = color_mean[i]

    for color in color_list:
        df[color] = color_ann_cumulation[color]

    fig_list = []
    for color in color_list:
        fig = px.box(
            df.sort_values(by="class_name"),
            x="class_name",
            y=color,
            color="class_name",
            color_discrete_sequence=LABEL_COLORS_WOUT_NO_FINDING,
            notched=True,
            labels={"class_name": "Class Name", "frac_bbox_area": "BBox Area (%)"},
            title="<b>Class 별 이미지 내 Bbox 의 " + color + " 분포 </b>",
        )
        fig.update_layout(
            showlegend=True,
            yaxis_range=[-10, 260],
            legend_title_text=None,
            xaxis_title="",
            yaxis_title="<b> " + color + " </b>",
        )
        fig_list.append(fig)

    with open(f"utils/distribution_plotly/{dir}_Color distribution", "wb") as fw:    
        pickle.dump(fig_list, fw)
        
def make_class_dist_figs(dir, df):
    return


def make_class_num_dist_figs(dir, df):
    return