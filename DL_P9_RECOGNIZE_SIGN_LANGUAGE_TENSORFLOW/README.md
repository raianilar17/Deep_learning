# RECOGNIZE_SIGN_LANGUAGE through TENSORFLOW(Deep Learning Framework)

 One midday, with some friends I decided to teach my computers to `decipher sign language`. 
 I spent a few hours taking pictures in front of a white wall and came up with the following dataset. 
 It's now my job to build an algorithm that would facilitate communications from a `speech-impaired person to someone who doesn't understand sign language`.

- **Training set**: 1080 pictures (64 by 64 pixels) of signs representing numbers from 0 to 5 (180 pictures per number).
- **Test set**: 120 pictures (64 by 64 pixels) of signs representing numbers from 0 to 5 (20 pictures per number).

Note that this is a subset of the SIGNS dataset. The complete dataset contains many more signs.

Here are examples for each number, and how an explanation of how I represent the labels. These are the original pictures, before I lowered the image resolutoion to 64 by 64 pixels.
<img src="images/hands.png" style="width:800px;height:350px;"><caption><center> <u><font color='purple'> **Figure 1**</u><font color='purple'>: SIGNS dataset <br> <font color='black'> </center>
 
 
## Goal:
 To build an algorithm capable of recognizing a sign with high accuracy. 
 To do so, I went to build a tensorflow model with using a softmax output.

The model is LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX. The SIGMOID output layer has been converted to a SOFTMAX.
A SOFTMAX layer generalizes SIGMOID to when there are more than two classes.

## Files:

- [datasets](datasets): sign datasets
- [images](images): test images
- [Sign_language_TensorFlow.ipynb](Sign_language_TensorFlow.ipynb): Complete project describtion
- [improv_utils.py](improv_utils.py): some useful functions
- [tf_utils.py](tf_utils.py): Some useful functions

## Commands to Run Code :
Run in jupyter notebook(The Jupyter Notebook is an open-source web application that allows you to create and share documents that contain live code, equations, visualizations and narrative text. Uses include: data cleaning and transformation, numerical simulation, statistical modeling, data visualization, machine learning, and much more.)

Detail explanation is given in notebook. [Please refer](Sign_language_TensorFlow.ipynb).

The notebook written in deep explanation with mathematical function.
 
 ## Accuracy:
 - Training_Accuracy : 99.99 %
 - Testing_Accuracy  : 81 %
 
My algorithm can recognize a sign representing a figure between 0 and 5 with 81% accuracy.



## Things to keep in mind:

- Tensorflow is a programming framework used in deep learning
- The two main object classes in tensorflow are `Tensors` and `Operators`. 
- When we code in tensorflow we have to take the following steps:
    - Create a graph containing Tensors (Variables, Placeholders ...) and Operations (tf.matmul, tf.add, ...)
    - Create a session
    - Initialize the session
    - Run the session to execute the graph
- we can execute the graph multiple times.
- The backpropagation and optimization is automatically done when running the session on the "optimizer" object.

### Future work:
- My model seems big enough to fit the training set well. 
However, given the difference between train and test accuracy, In future, I'll try to add L2 or dropout regularization to reduce overfitting. 
