# Emotion Detection in Images of Faces Using Keras

Emotion Tracking

* A nearby community health clinic is helping the local residents monitor their mental health.  
* As part of their study, They are asking volunteers to record their emotions throughout the day.
* To help the participants more easily track their emotions, I am asked to create an app that will classify their emotions based on some pictures that the volunteers will take of their facial expressions.
* As a proof-of-concept, I first train my model to detect if someone's emotion is classified as "happy" or "not happy."

To build and train this model, I have gathered pictures of some volunteers in a nearby neighborhood. The dataset is labeled.
<img src="images/face_images.png" style="width:550px;height:250px;">

## Files :
- [datasets](datasets)                         : Emotion datasets
- [images](images)                             : unseen images to test our model
- [HappyModel.png](HappyModel.png)             : model description
- [Keras_Tutorial.ipynb](Keras_Tutorial.ipynb) : Complete project desscription
- [kt_utils.py](kt_utils.py)                   : load datasets functions

## Commands to Run Code :
Run in jupyter notebook(The Jupyter Notebook is an open-source web application that allows you to create 
and share documents that contain live code, equations, visualizations and narrative text.
Uses include: data cleaning and transformation, numerical simulation, statistical modeling, data visualization, machine learning, and much more.)

Detail explanation is given in notebook. [Please refer](Keras_Tutorial.ipynb) .

The notebook written in deep explanation with mathematical function.

The code succesfully run on Keras

## Results :
- Training_Accuracy   : 96.67
- Validation_Accuracy : 98.33
- Testing_Accuracy    : 95.33

## Key Points to remember :

- Keras is a tool we recommend for rapid prototyping. It allows to quickly try out different model architectures.
-  Keras, a high-level neural networks API (programming framework),
written in Python and capable of running on top of several lower-level frameworks including TensorFlow and CNTK.
- Keras was developed to enable deep learning engineers to build and experiment with different models very quickly.
- Just as TensorFlow is a higher-level framework than Python, Keras is an even higher-level framework and provides additional abstractions.
- Being able to go from idea to result with the least possible delay is key to finding good models.
- However, Keras is more restrictive than the lower-level frameworks, so there are some very complex models 
that we would still implement in TensorFlow rather than in Keras.
- That being said, Keras will work fine for many common models.
- Remember The four steps in Keras:
  * Create
  * Compile
  * Fit/Train
  * Evaluate/Test

## Future work :
I will try to apply this model on different field.
