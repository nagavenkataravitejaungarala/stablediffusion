import streamlit as st
import requests
from PIL import Image
import io
from diffusers import StableDiffusionPipeline
import torch

# Streamlit app title
st.title("Stable Diffusion Image Generation")

# Set up the model
model_id = "dreamlike-art/dreamlike-photoreal-2.0"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Input box for user prompt
prompt = st.text_input("Enter your prompt")

# Button to generate the image
if st.button("Generate Image"):
    image = pipe(prompt).images[0]
    
    # Display the generated image
    st.image(image, use_column_width=True)

# Note to the user
st.markdown("NOTE: This app requires a GPU to run.")

