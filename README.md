# Multimodal-and-multicontrast-image-fusion-via-deep-generative-models
Repository of the paper "Multimodal and multicontrast image fusion via deep generative models"

When using this code please cite: Dimitri, Giovanna Maria, et al. "Multimodal and multicontrast image fusion via deep generative models." Information Fusion 88 (2022): 146-160.

Website for the paper: https://www.sciencedirect.com/science/article/pii/S1566253522000720

# Overview
We provide our experimentation code for predicting Multimodal and multicontrast image fusion via deep generative models.

Train_example.py runs our evaluation experiments
utils.py contains:
- Out data loading and preprocessing procedures (preprocess.py)
- The data iterator (augmentation.py)
- A custom implementation of 3D separable convolutions (sepconv3D.py)
- The network model architecture (models.py)

