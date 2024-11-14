"""
Image processing tasks for preparing and visualizing Pokémon images.
"""

import numpy as np
from skimage.segmentation import slic, mark_boundaries
import streamlit as st

def process_image(img, processor, device):
    """
    Prepares the input tensor for the model using the specified processor.
    Converts the image into model-friendly data that runs on the chosen device (CPU/GPU).
    """
    inputs = processor(images=img, return_tensors="pt").to(device)
    return inputs["pixel_values"]

@st.cache_data
def visualize_superpixels(img, n_segments, compactness):
    """
    Applies superpixel segmentation to show the underlying structure of the image.
    Displays the superpixel boundaries, so we can get a sense of how the model "sees" different parts of the image.
    """
    segments = slic(np.array(img), n_segments=n_segments, compactness=compactness)
    superpixel_image = mark_boundaries(np.array(img) / 255.0, segments)  # Normalize image for display
    st.subheader("Superpixels Segmentation")
    st.image(superpixel_image, caption="Superpixels")

# TODO: Consider adding more segmentation options to give users more control over visualization
# TODO: Possibly refactor to support other image transformations, if needed
