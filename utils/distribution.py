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

LABEL_COLORS = [
    px.colors.label_rgb(px.colors.convert_to_RGB_255(x))
    for x in sns.color_palette("Spectral", 10)
]
LABEL_COLORS_WOUT_NO_FINDING = LABEL_COLORS[:8] + LABEL_COLORS[9:]

def set_data(dir:str) -> pd.DataFrame:
    """데이터 설정

    Args:
        dir (str): json 파일 경로

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
    segmentations_ids = []
    
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
            segmentations_ids.append(ann["id"])
            areas.append(ann["area"])
            sizes.append(image_info["width"] * image_info["height"])
            

    df["image_id"] = image_ids
    df["category_id"] = category_ids
    df["category_name"] = category_names
    df["segmentation"] = segmentations
    df["area"] = areas
    df["size"] = sizes
    df["segmentation_id"] = segmentations_ids
    return df
    
def show_distribution(dir:str, category:str):
    """해당 폴더의 해당 카테고리의 distribution image를 반환

    Args:
        dir (str): distribution을 확인하려는 폴더 경로
        category (str): distribution을 확인하려는 카테고리
    """
    
    #if f"{dir}_{category}" not in os.listdir("utils/distribution_plotly") or st.button("refresh"):
    #make_dist_figs(dir, category)
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
    json_path = f"data/{dir}/train_all.json"
    df = set_data(json_path)
    if category == "Segmentation proportion distribution":
        make_prop_dist_figs(dir, df)
    elif category == "Color distribution":
        make_color_dist_figs(dir, df)
    elif category == "Class number distribution":
        make_class_dist_figs(dir, df)
    elif category == "Object number per image distribution":
        make_object_per_img_dist_figs(dir, df)
    

def make_prop_dist_figs(dir:str, df:pd.DataFrame):
    """annotation proportion distribution 계산 및 저장

    Args:
        dir (str): distribution을 확인하려는 폴더 경로
        df (pd.DataFrame) : coco format의 annotations들을 하나의 행으로 하는 데이터프레임
    """
    
    df = df.sort_values(["image_id", "category_id"])
    areas = dict()
    fig_list = []
    last_image_id = df.iloc[0,1]
    last_category_name = df.iloc[0,2]
    size = 0
    for row in df.itertuples():
        now_size = row.area
        if row.image_id == last_image_id and row.category_name == last_category_name:
            size += now_size
        else:    
            if last_category_name in areas.keys():
                areas[last_category_name].append(size / (512**2) * 100)
            else:
                areas[last_category_name] = [size / (512**2) * 100]
            size = now_size
            last_image_id = row.image_id
            last_category_name = row.category_name
    areas[row.category_name].append(size / (512**2) * 100)
    
    for area in areas.keys():
        fig = px.histogram(areas[area], labels = {"value" : "proportion(%)"}, title = area)
        fig.update_layout(
        title=dict(
            text=f'<b>{area}</b>',
            x=0.5,
            y=0.87,
            font=dict(
                family="Arial",
                size=40,
                color="#000000"
            )
        ),
        xaxis_title=dict(
            text="<b>Total segmentation proportion per image(%)</b>"
        ),
        yaxis_title="<b>Count</b>",
        font=dict(
            size=12,
        ),
        showlegend=False,
        margin = dict(l=10, r=10, b=10)
    )
        fig_list.append(fig)
        
    with open(f"utils/distribution_plotly/{dir}_Segmentation proportion distribution", "wb") as fw:    
         pickle.dump(fig_list, fw)
        
def get_segmentations(img_group : pd.DataFrame) -> List[list]:
    segmentations = []
    for _, row in img_group.iterrows():
        s_id, seg = row.segmentation_id, row.segmentation[0]
        segmentations.append([s_id, seg])
    return segmentations
        
    
def make_color_dist_figs(dir, df: pd.DataFrame):
    """color distribution의 box plot 계산 및 저장
    Args:
        dir (str): distribution을 확인하려는 폴더 경로
        df (pd.DataFrame) : coco format의 annotations들을 하나의 행으로 하는 데이터프레임
    """
    color_list = ["r_mean", "g_mean", "b_mean", "h_mean", "s_mean", "v_mean"]
    group = df.groupby("image_id")
    img_paths = list(group.groups.keys())
    len_df = len(df)
    color_ann_cumulation = {color: [0] * len_df for color in color_list}

    for img_path in stqdm(img_paths):
        img_bgr = cv2.imread(os.path.join(f"data/{dir}/", img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        semgnentations = get_segmentations(group.get_group(img_path))
        rgb = []
        hsv = []
        for s_id, seg in semgnentations:
            len_seg = len(seg)
            for i in range(0, len_seg, 2):
                x, y = map(round, seg[i:i+2])
                if x > 512:
                    x = 512
                elif y > 512:
                    y = 512
                rgb.append(img_rgb[y, x, :])
                hsv.append(img_hsv[y, x, :])

        rgb_mean = np.mean(rgb, axis = 0)
        hsv_mean = np.mean(hsv, axis = 0)
        color_mean = np.concatenate((rgb_mean, hsv_mean))
        for i, color in enumerate(color_list):
            try:
                color_ann_cumulation[color][s_id] = color_mean[i]
            except:
                print(s_id)
                print(len_df)
                break


    for color in color_list:
        df[color] = color_ann_cumulation[color]

    fig_list = []
    for color in color_list:
        fig = px.box(
            df.sort_values(by="category_name"),
            x="category_name",
            y=color,
            color="category_name",
            color_discrete_sequence=LABEL_COLORS_WOUT_NO_FINDING,
            notched=True,
            labels={"category_name": "Category Name", "frac_bbox_area": "BBox Area (%)"},
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
        
def make_class_dist_figs(dir:str, df:pd.DataFrame):
    """category distribution 계산 및 저장

    Args:
        dir (str): distribution을 확인하려는 폴더 경로
        df (pd.DataFrame) : coco format의 annotations들을 하나의 행으로 하는 데이터프레임
    """
    figs = []
    fig = px.histogram(df, x="category_name")
    fig.update_layout(
    title=dict(
        text='<b>Class number</b>',
        x=0.5,
        y=0.87,
        font=dict(
            family="Arial",
            size=40,
            color="#000000"
        )
    ),
    xaxis_title=dict(
        text="<b>Class name</b>"
    ),
    yaxis_title="<b>Count</b>",
    font=dict(
        size=12,
    ),
    showlegend=False,
    margin = dict(l=10, r=10, b=10)
)
    figs.append(fig)
    with open(f"utils/distribution_plotly/{dir}_Class number distribution", "wb") as fw:    
         pickle.dump(figs, fw)


def make_object_per_img_dist_figs(dir:str, df:pd.DataFrame):
    """이미지 별 annotation 개수의 distribution 계산 및 저장

    Args:
        dir (str): distribution을 확인하려는 폴더 경로
        df (pd.DataFrame) : coco format의 annotations들을 하나의 행으로 하는 데이터프레임
    """
    ann_nums_dict= dict(df["image_id"].value_counts())
    ann_nums_df = pd.DataFrame(pd.Series(ann_nums_dict)) 
    figs = []
    fig = px.histogram(pd.DataFrame(ann_nums_df))
    fig.update_layout(
    title=dict(
        text='<b>Object number per image</b>',
        x=0.5,
        y=0.87,
        font=dict(
            family="Arial",
            size=40,
            color="#000000"
        )
    ),
    xaxis_title=dict(
        text="<b>Object number</b>"
    ),
    yaxis_title="<b>Count</b>",
    font=dict(
        size=12,
    ),
    showlegend=False,
    margin = dict(l=10, r=10, b=10)
)
    figs.append(fig)
    with open(f"utils/distribution_plotly/{dir}_Object number per image distribution", "wb") as fw:    
         pickle.dump(figs, fw)