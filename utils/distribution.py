from pycocotools.coco import COCO
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
import pickle
import sys
from stqdm import stqdm
from collections import Counter
    
def show_distribution(dir:str, category:str):
    """해당 폴더의 해당 카테고리의 distribution image를 반환

    Args:
        dir (str): distribution을 확인하려는 폴더 경로
        category (str): distribution을 확인하려는 카테고리
    """
    if st.button("refresh"):
        make_dist_figs(dir, category)
    display_plotly_figs("dir/category")
        

def display_plotly_figs(figs_path: str):
    """plotly figures를 streamlit page에 표시하는 함수, plotly figure list가 저장된 pickle 파일을 불러옴
    Args:
        figs_path: pickle 파일 경로
    """
    try:
        with open(figs_path, "rb") as fr:
            dist_figs = pickle.load(fr)
    except Exception:
        sys.stderr.write("No file: %s\n" % figs_path)
        exit(1)
    for dist_fig in dist_figs:
        st.plotly_chart(dist_fig)


def make_dist_figs(dir:str, category:str):
    """해당 폴더의 해당 카테고리의 distribution plot를 계산 및 저장
    Args:
        dir (str): distribution을 확인하려는 폴더 경로
        category (str): distribution을 확인하려는 카테고리
    """
    #
    return