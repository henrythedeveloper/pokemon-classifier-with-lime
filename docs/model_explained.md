# Model Explained

This Pokémon classifier uses a Vision Transformer (ViT) model, which works great for image classification. The ViT model was trained on Pokémon images, with LIME explanations added to make predictions more transparent.

## Model Repository

The model is hosted on Hugging Face, so you don’t have to worry about manually downloading large files. When you run the app, it will automatically load the model from the Hugging Face repository at [HenryLeSD/pokemon-classifier-vit](https://huggingface.co/HenryLeSD/pokemon-classifier-vit).

## Why ViT?
The Vision Transformer (ViT) architecture is ideal for image classification because it focuses on image patches rather than individual pixels. This makes it great for capturing details in Pokémon images, giving reliable predictions.

## Dataset
The model currently has limited data on newer-generation Pokémon, so predictions might be less accurate for those. An update with additional training data is planned.

## Model Limitations
This model performs well on older Pokémon generations but may have reduced accuracy with newer species.
