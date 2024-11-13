# API Documentation

## image_processing.py
- `process_image(img, processor, device)`: Prepares the image tensor for the model.
- `visualize_superpixels(img, n_segments, compactness)`: Applies superpixel segmentation to visualize the image regions.

## lime_explainer.py
- `explain_with_lime(img, model, processor, device, top_indices, n_segments, compactness)`: Generates LIME explanations for given image predictions.
- `generate_textual_explanation(explanation, top_indices, class_labels, img)`: Creates text-based explanations for each top prediction.

## model.py
- `load_model()`: Loads the ViT model and processor directly from Hugging Face, along with class labels. It sets up the model for prediction, mapping indices to labels and vice versa.

Each function has been optimized for use in Streamlit and works with the ViT model.
