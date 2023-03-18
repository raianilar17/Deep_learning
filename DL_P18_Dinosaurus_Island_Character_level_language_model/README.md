# Dinosaurus_Island_Character_level_language_model

Welcome to Dinosaurus Island! 65 million years ago, dinosaurs existed, and In this Project they are back. 
We are in charge of a special task. Leading biology researchers are creating new breeds of dinosaurs and bringing them to life on earth, 
and our job is to give names to these dinosaurs. If a dinosaur does not like its name, it might go berserk, so choose wisely! 

<table>
<td>
<img src="images/dino.jpg" style="width:250;height:400px;">

</td>

</table>

Luckily I have learned some deep learning and I used it to save the day. Researchers has collected a list of all the dinosaur names they could find, 
and compiled them into this [dataset](dinos.txt). `To create new dinosaur names, I built a character level language model to generate new names`.
My algorithm will learn the different name patterns, and randomly generate new names. Hopefully this algorithm will keep my and mine team safe from 
the dinosaurs' wrath! 

By completing this project I learnt:

- How to store text data for processing using an RNN 
- How to synthesize data, by sampling predictions at each time step and passing it to the next RNN-cell unit
- How to build a character-level text generation recurrent neural network
- Why clipping the gradients is important

##  Overview of the model

My model have the following structure: 

- Initialize parameters 
- Run the optimization loop
    - Forward propagation to compute the loss function
    - Backward propagation to compute the gradients with respect to the loss function
    - Clip the gradients to avoid exploding gradients
    - Using the gradients, update our parameters with the gradient descent update rule.
- Return the learned parameters 
    
<img src="images/rnn.png" style="width:450;height:300px;">
<caption><center> Figure 1: Recurrent Neural Network, similar to what I had built in the previous project "Building a Recurrent Neural Network - Step by Step".  </center></caption>

* At each time-step, the RNN tries to predict what is the next character given the previous characters. 
* The dataset X = (x^<1>, x^<2> , ..., x^<T_x>) is a list of characters in the training set.
* Y = (y^<1>, y^<2>, ..., y^<T_x> is the same list of characters but shifted one character forward. 
* At every time-step t , y^<t> = x^< t+1 >.  The prediction at time t is the same as the input at time t + 1.

## Files:
- [images](images) : project related images
- [models](models) : pre-defined models
- [Dinosaurus_Island_Character_level_language_model_final.ipynb](Dinosaurus_Island_Character_level_language_model_final.ipynb): complete project description
- [dinos.txt](dinos.txt) : dinosaurus name
- [requirements.txt](requirements.txt) : install packages
- [shakespeare.txt](shakespeare.txt) : shakespeare datasets
- [shakespeare_utils.py](shakespeare_utils.py) : shakespeare datasets
- [utils.py](utils.py) : Pre-defined functions


## Commands to run code :
- install keras (pip install keras==2.0.8)
- Run jupyter notebook([Dinosaurus_Island_Character_level_language_model_final.ipynb](Dinosaurus_Island_Character_level_language_model_final.ipynb)) 
(The Jupyter Notebook is an open-source web application that allows us to create and share documents that contain live code, equations, 
visualizations and narrative text. Uses include: data cleaning and transformation, numerical simulation, statistical modeling, data visualization,
machine learning, and much more.)
- Detail explanation is given in notebook. Please refer [Dinosaurus_Island_Character_level_language_model_final.ipynb](Dinosaurus_Island_Character_level_language_model_final.ipynb) .
- The notebook written in deep explanation with mathematical function.
