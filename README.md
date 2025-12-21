# Autonomous Vehicle Steering Control via End-to-End Deep Learning in BeamNG.tech

## Overview
This project explores end-to-end deep learning for autonomous steering using the BeamNG.tech simulator. The goal is to predict steering angles directly from camera images in real-time.

## Features
- Predicts steering angles from front and side camera images
- Lane recovery using side camera offsets
- Three CNN architectures tested, including residual and attention mechanisms
- Data augmentation for more robust learning

## Models
- Implemented and trained 3 models each with different number of layers and structure
- Model 3 was the best I achieved which predicted angles better than the REST (but still not trust worthy_

## Training Details
- **Loss Function:** Mean Squared Error (MSE)
- **Optimizer:** Adam (LR=0.002) with weight decay
- **Data Augmentation:** Random flip, translation, brightness changes
- **Validation:** 25% of dataset reserved
- **Note:** Models were partially trained due to limited computation power, which affected performance

## Results




