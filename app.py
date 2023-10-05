import streamlit as st
from imgcaptioning.inference_pipeline import inference

st.set_page_config(
    page_title="Image Caption Generator",
    page_icon="💻",
    menu_items={
        'Report a bug': "https://github.com/KBVijayVarma/image-captioning/issues",
        'About': "Just upload image and generate captions for free"
    }
)

st.markdown(
    "<h1 style='text-align: center;'>Image Caption Generator</h1>",
    unsafe_allow_html=True
)

image = st.file_uploader("Upload a image", type=['png', 'jpg'])

if image is not None:
    output = inference(image=image)
    st.image(image)
    st.write(output)
