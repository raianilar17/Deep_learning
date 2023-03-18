# Build Planar data classifier with Different initialization

## Different Initialization

Advantages of choose specific Initialization:

- It helps to improve Deep Neural Networks

- Training neural network requires specifying an initial value of the weights. A well chosen initialization method will help good learning.

- In  notebook,I explain, How do we choose the initialization for a neural network. The different initializations lead to different results.

A well chosen initialization can:

- Speed up the convergence of gradient descent.
- Increase the odds of gradient descent converging to a lower training (and generalization) error.

I use a 3-layer neural network. Here are the initialization methods I experiment with:

- Zeros initialization -- setting initialization = "zeros" in the input argument.
- Random initialization -- setting initialization = "random" in the input argument. This initializes the weights to large random values.
- He initialization -- setting initialization = "he" in the input argument. 
  This initializes the weights to random values scaled according to a paper by He et al.,
  2015  [ICLR_res_paper](ICLR_res_paper.pdf), [arxiv_res_paper](arxiv_res_paper.pdf).

## Files:
1. [ICLR_res_paper.pdf](ICLR_res_paper.pdf): Research paper
2. [arxiv_res_paper.pdf](arxiv_res_paper.pdf): Research paper
3. [Initialization.ipynb](Initialization.ipynb): Complete Project description
4. [init_utils.py](init_utils.py): Some useful functions

## Command to Run code

Run in jupyter notebook(The Jupyter Notebook is an open-source web application that 
allows you to create and share documents that contain live code, equations, visualizations and narrative text. 
Uses include: data cleaning and transformation, numerical simulation, statistical modeling, data visualization, 
machine learning, and much more.)

Detail explanation is given in notebook. [Please refer](Initialization.ipynb)

The notebook written in deep explanation with mathematical function.

## Result Analysis:

I proposes three different types of initializations. For the same number of iterations and same hyperparameters the comparison is:

<table> 
   <tr>
       <td>
       Model
        </td>
        <td>
        Train accuracy
        </td>
        <td>
        Test accuracy
        </td>
        <td>
        Problem/Comment
        </td>
     </tr>
        <td>
        3-layer NN with zeros initialization
        </td>
        <td>
        50%
        </td>
        <td>
        50%
        </td>
        <td>
        fails to break symmetry
        </td>
    <tr>
        <td>
        3-layer NN with large random initialization
        </td>
        <td>
        83%
        </td>
        <td>
        86%
        </td>
        <td>
        too large weights 
        </td>
    </tr>
    <tr>
        <td>
        3-layer NN with He initialization
        </td>
        <td>
        99%
        </td>
         <td>
        96%
        </td>
        <td>
        recommended method
        </td>
    </tr>
</table> 
  
 

## conclusions:

- Different initializations lead to different results
- Random initialization is used to break symmetry and make sure different hidden units can learn different things
- Don't intialize to values that are too large
- He initialization works well for networks with ReLU activations.
