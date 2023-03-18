# Deep-Neural-Network-for-Image-Classification-Application

Build and apply a deep neural network to supervised learning.

I used the functions I'd implemented in the previous project to build a deep network, and apply it to `cat vs non-cat classification`. I improve in accuracy relative to my previous logistic regression implementation.

## Files:

1. [Datasets](datasets) : training/testing datasets
2. [images](images) : Unseen images
3. [Deep_Neural_Network_Application .ipynb](Deep_Neural_Network_Application.ipynb) : Complete Project description
4. [dnn_app_utils_v3.py](dnn_app_utils_v3.py) :  provides some the functions 

## Command to Run code

Run in jupyter notebook(The Jupyter Notebook is an open-source web application that 
allows you to create and share documents that contain live code, equations, visualizations and narrative text. 
Uses include: data cleaning and transformation, numerical simulation, statistical modeling, data visualization, 
machine learning, and much more.)

Detail explanation is given in notebook. [Please refer](Deep_Neural_Network_Application.ipynb)

The notebook written in deep explanation with mathematical function.

## Result Analysis

A few types of images the model tends to do poorly on include:

    Cat body in an unusual position
    Cat appears against a background of a similar color
    Unusual cat color and species
    Camera Angle
    Brightness of the picture
    Scale variation (cat is very large or small in image)

### Future Work

In future, I will try to obtain even higher accuracy by systematically searching for better hyperparameters (learning_rate, layers_dims, num_iterations, and others).
