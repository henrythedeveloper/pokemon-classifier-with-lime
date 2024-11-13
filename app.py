# app.py - Pokémon Classifier with LIME Explanations
# Upload a Pokémon image, and let’s see what happens!

import io
import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from skimage.segmentation import mark_boundaries
from utils.lime_explainer import generate_textual_explanation
from utils import load_model, explain_with_lime, process_image, visualize_superpixels, setup_logging
from stqdm import stqdm  # Making sure stqdm is imported

# Set up logging for troubleshooting
logger = setup_logging()

# Kick off the app with a title and brief description
st.title('Pokémon Classifier with LIME Explanations')
st.write('''
This app classifies an image of a Pokémon using a Vision Transformer model, and shows LIME explanations to highlight what influenced each prediction.
''')

# Load the model and labels
model, processor, class_labels, label2id = load_model()
if model is None or processor is None:
    st.error("Failed to load the model or processor.")
    st.stop()  # Stop everything if we can’t load the essentials

# Section for segmentation parameters
st.markdown('''
### Segmentation Parameters
- **Number of segments**: Controls the number of segments (or "superpixels") in the image. More segments mean more detail.
- **Compactness**: Balances color proximity with spatial proximity. Higher values make the segments more square-shaped.
''')
n_segments = st.slider('Number of segments', min_value=10, max_value=100, value=50)
compactness = st.slider('Compactness', min_value=1, max_value=100, value=30)

# Check if GPU is available; use CPU otherwise
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Upload image file
uploaded_file = st.file_uploader('Choose an image file', type=['png', 'jpg', 'jpeg'])
if uploaded_file is not None:
    try:
        # Open the uploaded image
        img = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
        st.image(img, caption="Uploaded Image")

        # Process the image for model input
        img_tensor = process_image(img, processor, device)

        # Show superpixel segmentation on the image
        visualize_superpixels(img, n_segments, compactness)

        # Make predictions - showing top 3 guesses
        with torch.no_grad():  # No gradients needed here
            logits = model(img_tensor).logits
        probabilities = F.softmax(logits, dim=1)[0].cpu().numpy()  # Convert logits to probabilities

        # Display the top 3 predicted Pokémon with probabilities
        top_indices = probabilities.argsort()[-3:][::-1]
        top_probs = [(class_labels[idx], probabilities[idx]) for idx in top_indices]

        st.subheader("Top 3 Predictions")
        data = {
            "Pokémon": [label for label, _ in top_probs],
            "Probability": [f"{prob:.4f}" for _, prob in top_probs]
        }
        st.dataframe(pd.DataFrame(data))  # Present predictions in a table

        # LIME explanations to highlight key features influencing the prediction
        explanation = explain_with_lime(img, model, processor, device, top_indices, n_segments, compactness)

        # Display LIME explanations visually for each prediction
        st.subheader("LIME Explanations")
        cols = st.columns(len(top_indices))
        for i, idx in enumerate(top_indices):
            if idx in explanation.local_exp:
                # Highlight important regions of the image
                temp, mask = explanation.get_image_and_mask(
                    label=idx,
                    positive_only=False,
                    num_features=10,
                    hide_rest=False
                )
                img_boundary = mark_boundaries(temp / 255.0, mask)
                cols[i].image(img_boundary, caption=f"{class_labels[idx]}")
            else:
                st.write(f"No LIME explanation found for {class_labels[idx]}")

        # Generate textual explanation with a loading spinner
        with st.spinner('Generating textual explanations...'):
            try:
                textual_explanation = generate_textual_explanation(
                    explanation,
                    top_indices,
                    class_labels,
                    img
                )
                st.write(textual_explanation)
            except Exception as e:
                st.error(f"An error occurred during explanation generation: {e}")
                logger.error(f"Error in generate_textual_explanation: {e}")

    except Exception as e:
        # Error handling for unexpected issues
        st.error(f"An error occurred: {e}")
        logger.error(f"Unexpected error: {e}")

# TODO: Add caching for model loading to save time when reloading
# TODO: Add more detailed error handling, especially for LIME explanations
# TODO: Update the model with newer Pokémon generations to improve accuracy
