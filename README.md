# Pokémon Classifier with LIME Explanations

A fun app that lets you upload Pokémon images and get predictions with explanations! This project uses a Vision Transformer (ViT) model to classify Pokémon images and provides interpretability using LIME to highlight the image regions that contribute to each prediction.explanations.

## Overview

This project is a hands-on look at machine learning, model interpretability, and web app development. Powered by a Vision Transformer, it recognizes Pokémon from images, providing LIME (Local Interpretable Model-Agnostic Explanations) visualizations that reveal which areas of the image contributed to the model's predictions.

Note: The model might get a bit iffy on newer-gen Pokémon (working on it!). But, hey, classics are always a safe bet.

## Features

- **Image Classification**: Upload a Pokémon image and get the model's best guess on its species.
- **LIME Explanations**: Highlights the areas in the image that the model focused on when making its predictions.
- **Interactive Web Interface**: Friendly and easy-to-use interface built with Streamlit.

## Installation
### Prerequisites

- Python 3.7 or higher

Clone this repo, jump into your virtual environment (Python or Anaconda is great), and install the packages:

```pip install -r requirements.txt```

## Usage

Fire up the app with:

```streamlit run app.py```

It'll open in your browser, and from there, just upload a pic of a Pokémon to see the classification and LIME magic in action.

## Demo

## Data Limitations

- The model might have lower accuracy on newer Pokémon due to limited data.
- Planning to retrain with more data soon to help it recognize newer species.

## Project Goals

This is more than just a classifier – it’s part of a class project where I dive into machine learning workflows and interpretability. Using LIME here is a cool way to show how you can make complex models more transparent.

## Contributing

Want to help? Here’s the plan:

1. Fork it.
2. Branch out for your feature or fix.
3. Commit and push it to your fork.
4. Open a pull request with the details.

## License

MIT License – check the [LICENSE](LICENSE) file for all the legal stuff.

## Acknowledgments

- [Hugging Face Transformers]([https://huggingface.co/google/vit-base-patch16-224-in21k]) -  For the ViT model and tools.
- [LIME](https://github.com/marcotcr/lime) - For making model explanations possible.
- [Streamlit](https://streamlit.io/) - For the fantastic web app framework.
- Shoutout to my instructor and classmates for the guidance and support!

Big shoutout to [Hugging Face Transformers]([https://huggingface.co/google/vit-base-patch16-224-in21k]) for the ViT model, [LIME](https://github.com/marcotcr/lime) for model explanations, and [Streamlit](https://streamlit.io/) for making interactive web apps easy. Also, thanks to my class instructor and classmates for the support.

---

## `docs/` Folder Guide
Here’s what to include in the `docs`/ folder so others can easily follow along or contribute:

1. `setup_guide.md`:
     - Step-by-step instructions on getting the project up and running.
      - Common troubleshooting tips (like resolving installation issues).

2. `model_explained.md`:
    - A rundown of the Vision Transformer (ViT) model: why we chose it, how it’s trained, and its limitations.
    - A note on the dataset used and why newer-gen Pokémon might not be recognized as well.

3. `lime_guide.md`:
    - Explanation of how LIME works, with a couple of examples.
    - Some pros and cons of using LIME with image data and the current segmentation settings.

4. `api_docs.md`:
    - Quick reference for each key function in the code. Include input/output details and what each function does.
    - Focus on the core modules (`image_processing.py`, `lime_explainer.py`, `model.py`).

5. `contributing.md`:
    - Tips for anyone who wants to contribute! Include a few guidelines for code style, testing, and creating branches.

6. `changelog.md`:
    - A running list of updates, fixes, and new features added to the project.

These docs will keep everything organized and help anyone interested in understanding, using, or contributing to the project. Let me know if you want help writing any of these files!

## Contact

Questions? Suggestions? Open an issue or shoot me an email at [hlsoftdev@proton.me].