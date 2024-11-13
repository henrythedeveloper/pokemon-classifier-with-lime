"""
utils module - a toolkit for everything from model loading to explanations.
"""

# Bring in key functions and classes from each submodule to keep imports clean
from .model import load_model
from .lime_explainer import explain_with_lime, generate_textual_explanation
from .image_processing import process_image, visualize_superpixels
from .logging_setup import setup_logging

# Now importing from utils is straightforward – no need to dig through submodules
