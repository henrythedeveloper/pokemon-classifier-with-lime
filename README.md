# Pokémon Classifier with LIME Explanations

This is a fun little app that lets you upload Pokémon images to get instant predictions with explanations! Using a Vision Transformer (ViT) model, it classifies Pokémon images and uses LIME to highlight which parts of the image contributed most to each prediction.

## Overview

This project dives into machine learning, interpretability, and web app development. Powered by a Vision Transformer, the app identifies Pokémon and uses LIME (Local Interpretable Model-Agnostic Explanations) to show which areas of the image influenced the model's decisions.

Note: The model might be a bit uncertain with newer-gen Pokémon, but it’s solid with the classics. More data and updates are in the works!

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

This will open the app in your browser. From there, just upload a Pokémon image, and the app will display the classification and explanations.

## Demo

[Streamlit App](https://pokemon-classifier.streamlit.app/)

## Data Limitations

- The model might have lower accuracy on newer Pokémon due to limited data.
- Planning to retrain with more data soon to help it recognize newer species.

## Project Goals

Beyond just being a classifier, this project is part of a class assignment exploring machine learning workflows and model interpretability. Using LIME here makes complex models more understandable and transparent.

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

These docs should make it easy to understand, use, and contribute to the project. Let me know if you need help with any of them!

## Contact

Questions? Suggestions? Open an issue or shoot me an email at [hlsoftdev@proton.me].
