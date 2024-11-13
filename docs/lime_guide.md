# LIME Explanation Guide

## What is LIME?
LIME (Local Interpretable Model-Agnostic Explanations) helps explain why a model makes certain predictions by highlighting areas in the image that contributed most to the classification.

## How LIME Works in this Project
- We use superpixels to create interpretable regions in the image.
- LIME then assigns importance to these regions to show what influenced the model's decisions.

## Example of LIME in Action

After you upload a Pokémon image, the app shows the top 3 predictions along with highlighted regions indicating which parts of the image influenced each prediction. This makes it easier to understand why the model guessed a particular Pokémon.

## Pros and Cons
- **Pros**: Great for visualizing model decisions.
- **Cons**: Can be computationally expensive and may sometimes highlight irrelevant regions.
