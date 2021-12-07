import cv2
import streamlit as st
import numpy as np

# st.title("First Streamlit Web Application")
# st.markdown("This simple web application renders \
# the uploaded images into grayscale mode.")
#
# method = st.sidebar.radio('Go To ->', options=['Webcam', 'Image'])
#
# st.sidebar.title("First Streamlit Web Application")
# st.sidebar.markdown("This simple web application renders \
# the uploaded images into grayscale mode.")
#
# uploaded_file=st.sidebar.file_uploader(label="Upload Image",\
# type=["jpg","jpeg","png"],key="i")
#
#
# if uploaded_file is not None:
#     file_bytes = np.asarray(bytearray(uploaded_file.read()),\
#     dtype=np.uint8)
#     image = cv2.imdecode(file_bytes, 1)
#     st.subheader("Grayscale Image")
#     st.image(image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY),width=400)
#

st.title("Neural Style Transfer")
# st.sidebar.title('Navigation')
# method = st.sidebar.radio('Go To ->', options=['Image', 'Webcam'])
# method = st.sidebar.radio('Go To ->', options=['Webcam', 'Image'])
st.sidebar.header('Options')

# style_model_name = st.sidebar.selectbox("Choose the style model: ")
# style_model_path = style_models_dict[style_model_name]

# model = get_model_from_path(style_model_path)

# if method == 'Image':
#     image_input(model)
#     image_input(style_model_name)
# else:
#     webcam_input(model)
#     webcam_input(style_model_name)