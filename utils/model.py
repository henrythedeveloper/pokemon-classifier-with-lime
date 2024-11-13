"""
Model loading module - takes care of loading the ViT model and class labels for Pokémon classification.
"""

import json
import streamlit as st
from transformers import ViTImageProcessor, ViTForImageClassification

@st.cache(allow_output_mutation=True)
def load_model():
    """
    Loads class labels and the Vision Transformer (ViT) model, with error handling to catch any issues.
    """
    # Load class labels
    try:
        with open("class_labels.json", "r") as f:
            raw_class_labels = json.load(f)
    except Exception as e:
        st.error(f"Error loading class labels: {e}")
        return None, None, None, None

    # Process class labels to map from index to label name
    class_labels = {int(k): v.split(" - ")[1] for k, v in raw_class_labels.items()}
    label2id = {v: k for k, v in class_labels.items()}  # Reverse mapping (label to index)

    # Load the model and processor
    try:
        model = ViTForImageClassification.from_pretrained("pokemon_classifier_vit_pytorch")
        processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    except Exception as e:
        st.error(f"Error loading model or processor: {e}")
        return None, None, None, None

    # Put the model in evaluation mode since we’re not training here
    model.eval()

    # Configure the model to recognize the loaded labels
    model.config.id2label = class_labels
    model.config.label2id = label2id

    return model, processor, class_labels, label2id

# TODO: Add a fallback option if the model file isn't found locally (e.g., prompt to download)
# TODO: Test different ViT models and see if we get a boost in accuracy
# TODO: Consider caching model weights to avoid reloading if they’re static
