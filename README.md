# Petal Color Classifier

## Summary

Train a simple image classifier to assign petal color to pictures of flowers using the public Oxford 102 Flowers dataset and transfer learning. Review the .ipynb files in notebooks for exploration of the Oxford 102 Flowers dataset, transfer learning of the Convolutional Neural Network, and evaluation of its performance.

## Repository Structure

    flower-color-classifier/

        ├── README.md
        ├── requirements.txt
        ├── data/
        │   └── flowers-102/           # only created by running the notebook files after cloning the repository
        │
        ├── notebooks/
        │   ├── 01_exploration.ipynb   # EDA + color label generation
        │   ├── 02_training.ipynb      # CNN transfer learning from ResNet18
        │   └── 03_evaluation.ipynb    # confusion matrix + example predictions vs. labels
        │
        ├── src/
        │   ├── data/
        │   │   └── dataset.py         # Python classes held outside the notebooks
        │   └── models/
        │       └── classifier.py
        │
        └── results/
            ├── training_diagnostics_static.png
            └── best_model_weights_static.pt    # Included as attachment in release only

## Background

Convolutional Neural Networks are a type of machine learning model that you can imagine working a bit like the multi-lens glasses from *National Treasure*. Each lens in the glasses reveals a different layer of hidden information, and stacking them together gradually brings the final picture into sharp focus. For CNNs, early layers detect very simple patterns, middle layers detect more complex shapes, and later layers assemble these clues into something recognizable (e.g., “this flower is mostly purple-blue”).

At the end of the network is a Softmax layer, which turns the model’s raw scores into a set of values between 0 and 1 that sum to 1. These numbers express how confident the model is in each possible category. The category with the highest score becomes the CNN’s prediction. Instead of hand-crafting the lenses ourselves, the CNN learns them automatically during training. The better it learns to tune each lens, the more accurately it can classify new images.

In this exercise, we begin with a set of glasses that were already made and simply fine-tune final lens for our specific purpose.
