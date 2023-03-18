# Convolutional-Model-application

- Implement a fully functioning ConvNet using TensorFlow
- Build and train a ConvNet in TensorFlow for a classification problem('SIGNS' Classification Problem)

As a reminder, the SIGNS dataset is a collection of 6 signs representing numbers from 0 to 5.

<img src="images/SIGNS.png" style="width:800px;height:300px;">

## Files :

- [datasets](datasets) : signs datasets
- [images](images) : images(test)
- [cnn_utils.py](cnn_utils.py): some useful functions
- [Convolution_model_Application.ipynb](Convolution_model_Application.ipynb) : complete project description

## Commands to Run Code :

Run in jupyter notebook(The Jupyter Notebook is an open-source web application that allows you to create and share documents that contain live code, equations, visualizations and narrative text. Uses include: data cleaning and transformation, numerical simulation, statistical modeling, data visualization, machine learning, and much more.)

Detail explanation is given in notebook. [Please refer]((Convolution_model_Application.ipynb)).

The notebook written in deep explanation with mathematical function.

The code succesfully run on Tensorflow(1.15.0)

## results :
- Train Accuracy: 99%
- Test Accuracy: 85%

## Future Work :

I built a model that recognizes SIGN language with almost 80% accuracy on the test set.
I will actually improve its accuracy by spending more time tuning the hyperparameters, or using regularization (as this model clearly has a high variance).
