"""
LIME explanations module - explains what the model "sees" and why it makes certain predictions.
"""

from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from stqdm import stqdm

def explain_with_lime(img, model, processor, device, top_indices, n_segments, compactness):
    """
    Generates LIME visual explanations for the specified image.
    Sets up the LIME explainer, handles model predictions, and applies segmentation.
    """
    explainer = lime_image.LimeImageExplainer()
    segmentation_fn = SegmentationAlgorithm('slic', n_segments=n_segments, compactness=compactness)

    # Helper function to make batch predictions on superpixel images
    def batch_predict(images):
        # Convert images to model input format
        inputs = processor(images=[Image.fromarray(im.astype(np.uint8)) for im in images], return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = F.softmax(logits, dim=1)
        return probs.cpu().numpy()

    # Generate explanation for each top prediction
    explanation = explainer.explain_instance(
        np.array(img),
        batch_predict,
        labels=top_indices,
        hide_color=0,
        num_samples=1000,
        segmentation_fn=segmentation_fn
    )

    return explanation

def generate_textual_explanation(explanation, top_indices, class_labels, img):
    """
    Generates textual explanations for each top prediction, with progress updates.
    Describes the regions of the image that contribute most to each prediction.
    """
    total_steps = len(top_indices)
    textual_explanations = []

    # Progress loop to generate explanations for each top prediction
    for top_idx in stqdm(top_indices, desc="Generating explanations", total=total_steps):
        superpixel_weights = explanation.local_exp[top_idx]
        sorted_superpixels = sorted(superpixel_weights, key=lambda x: -abs(x[1]))  # Sort by importance
        top_superpixels = [sp_idx for sp_idx, weight in sorted_superpixels[:5]]  # Take top 5

        segments = explanation.segments
        mask = np.isin(segments, top_superpixels)
        y_indices, x_indices = np.where(mask)

        if len(x_indices) > 0 and len(y_indices) > 0:
            # Calculate mean position of top superpixels to describe focus region
            x_mean = np.mean(x_indices)
            y_mean = np.mean(y_indices)

            height, width, _ = np.array(img).shape

            # Determine approximate location of focus
            horizontal_pos = (
                'left' if x_mean < width / 3 else
                'right' if x_mean > width * 2 / 3 else
                'center'
            )
            vertical_pos = (
                'top' if y_mean < height / 3 else
                'bottom' if y_mean > height * 2 / 3 else
                'center'
            )

            explanation_text = f"The model focuses on the {vertical_pos}-{horizontal_pos} region to predict {class_labels[top_idx]}."
        else:
            explanation_text = f"The model predicts {class_labels[top_idx]}, but key regions couldn't be determined."

        textual_explanations.append(explanation_text)

    return "\n\n".join(textual_explanations)

# TODO: Experiment with different segmentation algorithms in explain_with_lime for potentially better explanations
# TODO: Consider caching predictions in batch_predict for efficiency when testing
