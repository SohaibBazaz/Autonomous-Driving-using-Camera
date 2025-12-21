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
- Note: Car was purposely parked at an angle to check if it goes into lane
- 
## Training Phase:
<img width="499" height="296" alt="image" src="https://github.com/user-attachments/assets/85164024-3662-4f86-8767-a42a0775657c" />

## Model 1 Demo: (Baseline so it tries to maneuver towards the left but fails and crashes while steering right)


https://github.com/user-attachments/assets/107daaae-d2b0-4cb8-bd5e-f35f791b00c5



## Model 2 Demo: (Performed Slightly better than Model 1, i.e is able to steer its way towards the road trying to avoid the obstacles)


https://github.com/user-attachments/assets/3d9713a2-d861-4cef-a51e-133a0bab2aee



## Model 3 Demo: (Performed better than the previous ones. Although not perfect but still managed to go in a straight line from an angled parking postion)
