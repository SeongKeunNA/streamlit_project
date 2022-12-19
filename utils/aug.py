import streamlit as st
import json
import numpy as np

import albumentations as A

def get_placeholder_params(image: np.ndarray):
    return {
        "image_width": image.shape[1],
        "image_height": image.shape[0],
        "image_half_width": int(image.shape[1] / 2),
        "image_half_height": int(image.shape[0] / 2),
    }


@st.cache
def load_augmentations_config(
    placeholder_params: dict, path_to_config: str = "configs/augmentations.json"
) -> dict:
    """Load the json config with params of all transforms
    Args:
        placeholder_params (dict): dict with values of placeholders
        path_to_config (str): path to the json config file
    """
    with open(path_to_config, "r") as config_file:
        augmentations = json.load(config_file)
    for name, params in augmentations.items():
        params = [fill_placeholders(param, placeholder_params) for param in params]
    return augmentations


def fill_placeholders(params: dict, placeholder_params: dict) -> dict:
    """Fill the placeholder values in the config file
    Args:
        params (dict): original params dict with placeholders
        placeholder_params (dict): dict with values of placeholders
    """
    # TODO: refactor
    if "placeholder" in params:
        placeholder_dict = params["placeholder"]
        for k, v in placeholder_dict.items():
            if isinstance(v, list):
                params[k] = []
                for element in v:
                    if element in placeholder_params:
                        params[k].append(placeholder_params[element])
                    else:
                        params[k].append(element)
            else:
                if v in placeholder_params:
                    params[k] = placeholder_params[v]
                else:
                    params[k] = v
        params.pop("placeholder")
    return params


def select_transformations(augmentations: dict) -> list:
    # in the Simple mode you can choose only one transform
    transform_names = [
        st.sidebar.selectbox(
            "Select transformation №1:", sorted(list(augmentations.keys())), label_visibility="hidden",
        )
    ]
    while transform_names[-1] != "None":
        transform_names.append(
            st.sidebar.selectbox(
                f"Select transformation №{len(transform_names) + 1}:",
                ["None"] + sorted(list(augmentations.keys())),
                label_visibility="hidden",
            )
        )
    transform_names = transform_names[:-1]

    return transform_names

def get_transormations_params(transform_names: list, augmentations: dict) -> list:
    transforms = []
    for i, transform_name in enumerate(transform_names):
        # select the params values
        st.sidebar.subheader("Params of the " + transform_name)
        param_values = show_transform_control(augmentations[transform_name], i)
        transforms.append(getattr(A, transform_name)(**param_values))
    return transforms

def show_transform_control(transform_params: dict, n_for_hash: int) -> dict:
    param_values = {"p": 1.0}
    if len(transform_params) == 0:
        st.sidebar.text("Transform has no parameters")
    else:
        for param in transform_params:
            control_function = param2func[param["type"]]
            if isinstance(param["param_name"], list):
                returned_values = control_function(**param, n_for_hash=n_for_hash)
                for name, value in zip(param["param_name"], returned_values):
                    param_values[name] = value
            else:
                param_values[param["param_name"]] = control_function(
                    **param, n_for_hash=n_for_hash
                )
    return param_values


def select_num_interval(
    param_name: str, limits_list: list, defaults, n_for_hash, **kwargs
):
    st.sidebar.subheader(param_name)
    min_max_interval = st.sidebar.slider(
        param_name,
        limits_list[0],
        limits_list[1],
        defaults,
        key=hash(param_name + str(n_for_hash)),
    )
    return min_max_interval

def select_several_nums(
    param_name, subparam_names, limits_list, defaults_list, n_for_hash, **kwargs
):
    st.sidebar.subheader(param_name)
    result = []
    assert len(limits_list) == len(defaults_list)
    assert len(subparam_names) == len(defaults_list)

    for name, limits, defaults in zip(subparam_names, limits_list, defaults_list):
        result.append(
            st.sidebar.slider(
                name,
                limits[0],
                limits[1],
                defaults,
                key=hash(param_name + name + str(n_for_hash)),
            )
        )
    return tuple(result)


def select_min_max(
    param_name, limits_list, defaults_list, n_for_hash, min_diff=0, **kwargs
):
    assert len(param_name) == 2
    result = list(
        select_num_interval(
            " & ".join(param_name), limits_list, defaults_list, n_for_hash
        )
    )
    if result[1] - result[0] < min_diff:
        diff = min_diff - result[1] + result[0]
        if result[1] + diff <= limits_list[1]:
            result[1] = result[1] + diff
        elif result[0] - diff >= limits_list[0]:
            result[0] = result[0] - diff
        else:
            result = limits_list
    return tuple(result)


def select_RGB(param_name, n_for_hash, **kwargs):
    result = select_several_nums(
        param_name,
        subparam_names=["Red", "Green", "Blue"],
        limits_list=[[0, 255], [0, 255], [0, 255]],
        defaults_list=[0, 0, 0],
        n_for_hash=n_for_hash,
    )
    return tuple(result)

def replace_none(string):
    if string == "None":
        return None
    else:
        return string


def select_radio(param_name, options_list, n_for_hash, **kwargs):
    st.sidebar.subheader(param_name)
    result = st.sidebar.radio(param_name, options_list, key=hash(param_name + str(n_for_hash)))
    return replace_none(result)


def select_checkbox(param_name, defaults, n_for_hash, **kwargs):
    st.sidebar.subheader(param_name)
    result = st.sidebar.checkbox(
        "True", defaults, key=hash(param_name + str(n_for_hash))
    )
    return result

param2func = {
    "num_interval": select_num_interval,
    "several_nums": select_several_nums,
    "radio": select_radio,
    "rgb": select_RGB,
    "checkbox": select_checkbox,
    "min_max": select_min_max,
}

def show_random_params(data: dict):
    """Shows random params used for transformation (from A.ReplayCompose)"""
    st.subheader("Random params used")
    random_values = {}
    for applied_params in data["replay"]["transforms"]:
        random_values[
            applied_params["__class_fullname__"].split(".")[-1]
        ] = applied_params["params"]
    st.write(random_values)


def show_docstring(obj_with_ds):
    st.markdown("* * *")
    st.subheader("Docstring for " + obj_with_ds.__class__.__name__)
    st.text(obj_with_ds.__doc__)

def show_credentials():
    st.markdown("* * *")
    st.subheader("Credentials:")
    st.markdown(
        (
            "Source: [github.com/IliaLarchenko/albumentations-demo]"
            "(https://github.com/IliaLarchenko/albumentations-demo)"
        )
    )
    st.markdown(
        (
            "Albumentations library: [github.com/albumentations-team/albumentations]"
            "(https://github.com/albumentations-team/albumentations)"
        )
    )