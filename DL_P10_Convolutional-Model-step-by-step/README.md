# Convolutional-Model-step-by-step

## Outline of project :

I implemented the building blocks of a convolutional neural network! Each function I implement will have detailed instructions
that will walk you through the steps needed:

- Convolution functions, including:
    - Zero Padding
    - Convolve window 
    - Convolution forward
    - Convolution backward 
- Pooling functions, including:
    - Pooling forward
    - Create mask 
    - Distribute value
    - Pooling backward 
    
In This notebook I implement these functions from scratch in `numpy`. 
In the next Project, I use the TensorFlow equivalents of these functions to build the following model:

<img src="images/model.png" style="width:800px;height:300px;">

**Note** that for every forward function, there is its corresponding backward equivalent. 
Hence, at every step of your forward module I store some parameters in a cache. These parameters are used to compute gradients during backpropagation. 

## Files :

- [datasets](datasets) : datasets
- [images](images) : images related to project
- [Convolution_model_Step_by_Step.ipynb](Convolution_model_Step_by_Step.ipynb) : complete project description

## Commands to Run Code :

Run in jupyter notebook(The Jupyter Notebook is an open-source web application that allows you to create and share documents that contain live code, equations, visualizations and narrative text. Uses include: data cleaning and transformation, numerical simulation, statistical modeling, data visualization, machine learning, and much more.)

Detail explanation is given in notebook. [Please refer]((Convolution_model_Step_by_Step.ipynb)).

The notebook written in deep explanation with mathematical function.


## Future work

In future, I will use this functions to bulid robust applications
