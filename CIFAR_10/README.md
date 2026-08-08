# CIFAR 10 claases Classification using PyTorch

This project demonstrates how to build neural network using **PyTorch** to classify different 10 classes from the **CIFAR10 dataset**.

---
The project covers the complete deep learning workflow for CNN's in total 4 phases.

# Phase1 : Explore dataset and Simple CNN model
In this phase i learned:
- how to read data and why we normalize data. 
- visualize some images and understand the data structure.
- How to build base line cnn from scratch.
I got 76.97% Test accuracy in this phase. so, I believed that i need to improve model architecture.
# Phase 2: Improvement of CNN architecture
- For generalization i added augmentation
- Implemented **stacking convolution** by reading AlexNet paper
- Added batch normalization and Dropout and learning scheduler
I got 83.27% of Test accuracy and 81.90% Train accuracy
#  Phase 3: Transfer learning without fine tunning
- I used ResNet50 model ImageNet weights.
- I only trained about 5 epochs. But got 81.91% Test accuracy and 76.59% Train accuracy.
I observed that By only freezing last layer the model unable to recognize my images.
Because ResNet was imagnet data and it has 1000 claases.
# Phase 4: Transfer learning with fine tunning
- I unfreeze the Fully connected layer as well as layer4. so, the model is recogizes my claases high-level features. 
- I used 2 different learning rates. One is for layer 4 which is about 0.005, and another one is for FC layer which is 0.004. I used low learning rate for layer4, because The pretrained weights are already good I don't want to destroy them with big updates
- I got 90.58% Train accuracy and 89.54% test accuracy for only 5 epochs.
---
## Libraries Used

- PyTorch
- TorchVision
- Matplotlib
---
## Output

- Successfully trains a neural network on the CIFAR10 dataset.
- Achieves approximately **90.58% Train accuracy and 89.54% test accuracy**.
- Saves the trained model for future inference.
---
