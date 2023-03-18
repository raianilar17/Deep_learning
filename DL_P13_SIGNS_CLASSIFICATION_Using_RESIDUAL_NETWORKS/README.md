# DL_P13_SIGNS_CLASSIFICATION_Using_RESIDUAL_NETWORKS

In this project, I built very deep convolutional networks, using Residual Networks (ResNets). 
In theory, very deep networks can represent very complex functions; but in practice, they are hard to train. 
Residual Networks, introduced by [He et al.](https://arxiv.org/pdf/1512.03385.pdf),
allow we to train much deeper networks than were previously practically feasible.

**Outline of project :**
- I Implement the basic building blocks of ResNets. 
- I Put together these building blocks to implement and train a state-of-the-art neural network for image classification(signs datasets). 

**The problem of very deep neural networks :**

Last Project,I built different convolutional neural network. In recent years, neural networks have become deeper, with state-of-the-art networks going from just a few layers (e.g., AlexNet) to over a hundred layers.

* The main benefit of a very deep network is that it can represent very complex functions. It can also learn features at many different levels of abstraction, from edges (at the shallower layers, closer to the input) to very complex features (at the deeper layers, closer to the output). 
* However, using a deeper network doesn't always help. A huge barrier to training them is vanishing gradients: very deep networks often have a gradient signal that goes to zero quickly, thus making gradient descent prohibitively slow. 
* More specifically, during gradient descent, as we backprop from the final layer back to the first layer, We are multiplying by the weight matrix on each step, and thus the gradient can decrease exponentially quickly to zero (or, in rare cases, grow exponentially quickly and "explode" to take very large values). 
* During training, We might therefore see the magnitude (or norm) of the gradient for the shallower layers decrease to zero very rapidly as training proceeds: 

<img src="images/vanishing_grad_kiank.png" style="width:450px;height:220px;">
<caption><center> <u> <font color='purple'> Figure 1  </u><font color='purple'>  : Vanishing gradient <br> The speed of learning decreases very rapidly for the shallower layers as the network trains </center></caption>

I solved this problem by building a Residual Network!

## Building a Residual Network

In ResNets, a "shortcut" or a "skip connection" allows the model to skip layers:  

<img src="images/skip_connection_kiank.png" style="width:650px;height:200px;">
<caption><center> <u> <font color='purple'> Figure 2 </u><font color='purple'>  : A ResNet block showing a skip-connection <br> </center></caption>

The image on the left shows the "main path" through the network. The image on the right adds a shortcut to the main path. By stacking these ResNet blocks on top of each other, We can form a very deep network. 

The ResNet blocks with the shortcut also makes it very easy for one of the blocks to learn an identity function. This means that we can stack on additional ResNet blocks with little risk of harming training set performance.  
    
(There is also some evidence that the ease of learning an identity function accounts for ResNets' remarkable performance even more so than skip connections helping with vanishing gradients).

Two main types of blocks are used in a ResNet, depending mainly on whether the input/output dimensions are same or different. 

I implemented both of them: the "identity block" and the "convolutional block."

## The identity block

The identity block is the standard block used in ResNets, and corresponds to the case where the input activation (say a^{[l]}) has the same dimension as the output activation (say a^{[l+2]}). To flesh out the different steps of what happens in a ResNet's identity block, here is an alternative diagram showing the individual steps:

<img src="images/idblock2_kiank.png" style="width:650px;height:150px;">
<caption><center> <u> <font color='purple'> Figure 3 </u><font color='purple'>  : Identity block. Skip connection "skips over" 2 layers. </center></caption>

The upper path is the "shortcut path." The lower path is the "main path." In this diagram, we have also made explicit the CONV2D and ReLU steps in each layer. To speed up training we have also added a BatchNorm step. 

In this problem, I actually implemented a slightly more powerful version of this identity block, in which the skip connection "skips over" 3 hidden layers rather than 2 layers. It looks like this: 

<img src="images/idblock3_kiank.png" style="width:650px;height:150px;">
<caption><center> <u> <font color='purple'> Figure 4 </u><font color='purple'>  : Identity block. Skip connection "skips over" 3 layers.</center></caption>

## The convolutional block

The ResNet "convolutional block" is the second block type. We can use this type of block when the input and output dimensions don't match up. The difference with the identity block is that there is a CONV2D layer in the shortcut path: 

<img src="images/convblock_kiank.png" style="width:650px;height:150px;">
<caption><center> <u> <font color='purple'> Figure 4 </u><font color='purple'>  : Convolutional block </center></caption>

* The CONV2D layer in the shortcut path is used to resize the input x to a different dimension, so that the dimensions match up in the final addition needed to add the shortcut value back to the main path. 
* For example, to reduce the activation dimensions's height and width by a factor of 2, We can use a 1x1 convolution with a stride of 2. 
* The CONV2D layer on the shortcut path does not use any non-linear activation function. Its main role is to just apply a (learned) linear function that reduces the dimension of the input, so that the dimensions match up for the later addition step. 


## Files :

- [Research_paper](Research_paper)                   : Research papers of different CNN Architectures 
- [datasets](datasets)                               : signs datasets
- [images](images)                                   : unseen images
- [Residual_Networks.ipynb](Residual_Networks.ipynb) : Complete project description
- [model.png](model.png)                             : model description
- [resnets_utils.py](resnets_utils.py)               : Useful function

## Commands to Run Code :
Run in jupyter notebook(The Jupyter Notebook is an open-source web application that allows you to create and share documents that contain live code, equations, visualizations and narrative text. Uses include: data cleaning and transformation, numerical simulation, statistical modeling, data visualization, machine learning, and much more.)

Detail explanation is given in notebook. [Please refer](Residual_Networks.ipynb) .

The notebook written in deep explanation with mathematical function.

The code succesfully run on Keras

## Results :
- Training_Accuracy : 98.66
- Validation_Accuracy : 90.74
- Testing_Accuracy : 97.50

## KeyPoints to remember:

- Very deep "plain" networks don't work in practice because they are hard to train due to vanishing gradients.  
- The skip-connections help to address the Vanishing Gradient problem. They also make it easy for a ResNet block to learn an identity function. 
- There are two main types of blocks: The identity block and the convolutional block. 
- Very deep Residual Networks are built by stacking these blocks together.


### References 

This notebook presents the ResNet algorithm due to He et al. (2015). 
The implementation here also took significant inspiration and follows the structure given in the GitHub repository of Francois Chollet: 

- Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun - [Deep Residual Learning for Image Recognition (2015)](https://arxiv.org/abs/1512.03385)
- Francois Chollet's GitHub repository: https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py




