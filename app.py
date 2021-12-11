import streamlit as st
import pandas as pd
import numpy as np
import torch
from PIL import Image, ImageChops
import os
from torch.nn.functional import cross_entropy
from streamlit_image_comparison import image_comparison


st.set_page_config(layout="wide")



@st.cache(allow_output_mutation=True)
def load_model():
    efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
    return efficientnet.eval()
    
@st.cache(allow_output_mutation=True)
def load_classnames():
    with open("classes.txt") as file:
        return eval(file.read())

@st.cache(allow_output_mutation=True)
def load_images():
    files = os.listdir("./images")
    img_suffixes = ("jpg", "jpeg", "png")
    img_files = (f for f in files if f.endswith(img_suffixes))
    return [Image.open("./images/"+file) for file in img_files]

@st.cache(allow_output_mutation=True)
def load_styles():
    with open("style.css") as f:
        return '<style>{}</style>'.format(f.read())


st.markdown(load_styles(), unsafe_allow_html=True)


def img2tensor(img: Image) -> torch.Tensor:
    arr = np.array(img).transpose(2, 0, 1)[np.newaxis, ...]
    return torch.tensor(arr).float() / 255


def tensor2img(tensor: torch.Tensor) -> Image:
    tensor = tensor.squeeze(0) * 255
    arr = np.uint8(tensor.numpy()).transpose(1, 2, 0)
    return Image.fromarray(arr)


classnames = load_classnames()
images = load_images()
model = load_model()


if "selected_img" not in st.session_state:
    st.session_state["selected_img"] = images[0]

uploaded_file = st.sidebar.file_uploader("", type=['png', 'jpg', "jpeg"])
if uploaded_file is not None:
    uploaded_img = Image.open(uploaded_file)
    clicked = st.sidebar.button("analyze uploaded", key=100)
    if clicked:
        st.session_state.selected_img = uploaded_img

st.sidebar.markdown("<hr />", unsafe_allow_html=True) 
st.sidebar.markdown("or select from a few examples")

for i, img in enumerate(images):
    st.sidebar.markdown("<hr />", unsafe_allow_html=True) 
    st.sidebar.image(img)
    clicked = st.sidebar.button("analyze", key=i)
    if clicked:
        st.session_state.selected_img = img

st.sidebar.markdown("<hr />", unsafe_allow_html=True) 
st.sidebar.markdown("Photos source: "
    "<a href='https://unsplash.com/photos/pk_1RdcAfbE'>street sign</a>, "
    "<a href='https://unsplash.com/photos/X63FTIZFbZo'>clock on nightstand</a>, "
    "<a href='https://unsplash.com/photos/fAz5Cf1ajPM'>wine</a>, "
    "<a href='https://unsplash.com/photos/eWqOgJ-lfiI'>red cabin</a>, ",
    unsafe_allow_html=True)


top_k = 3
st.slider(min_value=0,
                   max_value=40,
                   label="sensitivity:",
                   value=20,
                   step=4,
                   key="slider")

@st.cache(allow_output_mutation=True)
def process(img):
    img_small = img.resize((300, 300), resample=Image.BILINEAR)
    input_tensor = img2tensor(img_small).repeat(top_k, 1, 1, 1)
    input_tensor.requires_grad = True
    prediction = model(input_tensor)
    confidences = torch.softmax(prediction.detach()[0], dim=-1)
    tops = torch.topk(confidences.flatten(), top_k)
    indeces = tops.indices.tolist()
    values = tops.values.tolist()
    target = torch.tensor(indeces)
    cross_entropy(prediction, target).backward()
    expl_tensors = [torch.mean(input_tensor.grad[option], axis=0, keepdim=True) for option in range(top_k)]
    return indeces, values, expl_tensors


img = st.session_state.selected_img
indeces, values, expl_tensors = process(img)

def label_formatter(i):
    index = indeces[i]
    confidence = values[i]
    return f"{classnames[index]} ({confidence*100:>.0f}%)"

option = st.radio("most likely objects in image:", options=range(top_k), format_func=label_formatter)
st.checkbox("blend explanation with image", key="blend")

expl_tensor = torch.abs(expl_tensors[option] * st.session_state.slider).clamp(0, 1).repeat(3, 1, 1)
expl_img = tensor2img(expl_tensor).resize(img.size)

if st.session_state.blend:
    expl_img = ImageChops.multiply(img, expl_img)

image_comparison(img, expl_img, in_memory=True)
