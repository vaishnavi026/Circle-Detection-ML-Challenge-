# Circle Detection ML Challenge:: 

## **Goal**

The objective of the Circle Detection ML Challenge is to design a robust circle detector capable of accurately locating circles in images with varying levels of noise. The challenge involves developing a deep learning model that takes an image as input and outputs the circle's center coordinates (x, y) and its radius.

## **Project Overview**

### Task:

Develop a deep-learning model to detect circles in images. The model should be trained to handle noisy images and provide precise estimates of the circle's parameters.

#### _Input_ <br/>
Input images with varying levels of noise.

#### _Output_ <br/>
Predicted location of the circle's center (x, y).\
The predicted radius of the detected circle.

### Dataset

I've generated a custom dataset with the following characteristics:\
Size: 40,000 images (tried on smaller dataset, but did not get good results)\
Train-Test Split: 80% for training, 20% for testing\
Noise Levels: Varied from 0.1 to 0.5 to simulate real-world scenarios.\

#### Data Generation

I've implemented the data generation process in prepare_data.py, which includes functionality for creating samples with varying noise parameters and splitting the dataset into training and testing sets.

Image Characteristics\
Color Space: Grayscale (1 channel) for simplicity in model architecture.\
Noise Parameters: Varied to ensure the model's adaptability to different noise levels.
