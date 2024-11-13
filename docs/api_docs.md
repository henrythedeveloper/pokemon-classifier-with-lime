# API Documentation

## image_processing.py
- `process_image(img, processor, device)`: Prepares the image tensor for the model.
- `visualize_superpixels(img, n_segments, compactness)`: Applies superpixel segmentation to visualize the image regions.

## lime_explainer.py
- `explain_with_lime(img, model, processor, device, top_indices, n_segments, compactness)`: Generates LIME explanations for given image predictions.
- `generate_textual_explanation(explanation, top_indices, class_labels, img)`: Creates text-based explanations for each top prediction.

## model.py
- `load_model()`: Loads the ViT model and class labels, configuring them for classification.

Each function has been optimized for use in Streamlit and works with the ViT model.
