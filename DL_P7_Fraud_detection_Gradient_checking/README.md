# Fraud_detection_Gradient_checking

Build a deep learning model to detect fraud--whenever someone makes a mobile payment with Gradient Checking.

In this project I implement and use gradient checking.

The objective of implementing gradient checking is to make sure that backpropagation implementation is correct.

Suppose,There is a company of mobile payments available globally, 
and are asked to build a deep learning model to detect fraud--whenever someone makes a payment, 
They want to see if the payment might be fraudulent, such as if the user's account has been taken over by a hacker.

But backpropagation is quite challenging to implement, and sometimes has bugs.
Because this is a mission-critical application, The implementation of backpropagation should be correct. 
To give this reassurance, I am going to use "gradient checking" for a proof that backpropagation is actually working!.

## Files
- [images](images): projects related images
- [Gradient_Checking.ipynb](Gradient_Checking.ipynb): Complete Project Description
- [gc_utils.py](gc_utils.py): Pre-defined function
- [testCases.py](testCases.py): testcases for verify the functions

## Commands to Run Code

Run in jupyter notebook(The Jupyter Notebook is an open-source web application that allows you to create and share documents that contain live code, equations, visualizations and narrative text. Uses include: data cleaning and transformation, numerical simulation, statistical modeling, data visualization, machine learning, and much more.)

Detail explanation is given in notebook. [Please refer](Gradient_Checking.ipynb).

The notebook written in deep explanation with mathematical function.

## Conclusions

- Gradient checking verifies closeness between the gradients from backpropagation and 
  the numerical approximation of the gradient (computed using forward propagation).
- Gradient checking is slow, so we don't run it in every iteration of training. 
 You would usually run it only to make sure your code is correct, 
 then turn it off and use backprop for the actual learning process. 
- Gradient Checking, doesn't work with dropout.
 You would usually run the gradient check algorithm without dropout to make sure my backprop is correct, then add dropout.
