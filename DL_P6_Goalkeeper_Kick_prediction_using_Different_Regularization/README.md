
## Build a deep learning model to find the positions on the field where the goalkeeper should kick the ball with using different Regularization.

Deep Learning models have so much flexibility and capacity that overfitting can be a serious problem, 
if the training dataset is not big enough. Sure it does well on the training set, 
but the learned network doesn't generalize to new examples that it has never seen!

I Use different regularization to build different deep learning models.

## Problem Statement: 
The French Football Corporation would like to recommend positions where France's goal keeper should kick the ball 
so that the French team's players can then hit it with their head. 

<img src="images/field_kiank.png" style="width:600px;height:350px;">
<caption><center> <u>Figure 1 </u>: Football field <br> The goal keeper kicks the ball in the air, the players of each team are fighting to hit the ball with their head </center></caption>


We have  2D dataset from France's past 10 games.

## Goal: Build a deep learning model to find the positions on the field where the goalkeeper should kick the ball.

Analysis of the dataset: 
This dataset is a little noisy, 
but it looks like a diagonal line separating the upper left half (blue) from the lower right half (red) would work well.

I first try a non-regularized model. Second ,I try different regularization technique and decide which model I 
choose to solve the French Football Corporation's problem.

## Files:
- [datasets](datasets): project datasets
- [images](images):images related to project
- [JML_res_paper.pdf](JML_res_paper.pdf): dropout related research paper
- [Regularization.ipynb](Regularization.ipynb): complete project desrciption
- [reg_utils.py](reg_utils.py): some useful functions
- [estCases.py](estCases.py): some useful function

## Command to Run code
Run in jupyter notebook(The Jupyter Notebook is an open-source web application that allows you to create and share documents that contain live code, equations, visualizations and narrative text. Uses include: data cleaning and transformation, numerical simulation, statistical modeling, data visualization, machine learning, and much more.)

Detail explanation is given in notebook. Please [refer](Regularization.ipynb)

The notebook written in deep explanation with mathematical function.

## Different regularization Technique:

1. non-regularized model 
    - non-regularized model is suffering from overfitting in the training set. 
    
2. L2 Regularization
   
   - The standard way to avoid overfitting is called L2 regularization.
   - The value of lambda(λ)  is a hyperparameter that tune using a dev set.
   - L2 regularization makes  decision boundary smoother.If  lambda(λ)  is too large, it is also possible to "oversmooth", resulting in a model with high bias.
   - L2-regularization relies on the assumption that a model with small weights is simpler than a model with large weights. Thus, by penalizing the square values of the weights in the cost function you drive all the weights to smaller values. It becomes too costly for the cost to have large weights! This leads to a smoother model in which the output changes more slowly as the input changes.
   
3. Dropout
    - Finally, dropout is a widely used regularization technique that is specific to deep learning. 
    - It randomly shuts down some neurons in each iteration.  
    - The idea behind drop-out is that at each iteration,
      train a different model that uses only a subset of your neurons.
      With dropout,neurons thus become less sensitive to the activation of one other specific neuron,
      because that other neuron might be shut down at any time.
    - A common mistake when using dropout is to use it both in training and testing. 
      You should use dropout (randomly eliminate nodes) only in training.  

## Key Points

- Regularization will help reduce overfitting.
- Regularization will drive our weights to lower values.
- L2 regularization and Dropout are two very effective regularization techniques.

## Conclusions

**Here are the results of our three models**: 

<table> 
    <tr>
        <td>
        Model
        </td>
        <td>
        Train Accuracy
        </td>
        <td>
        Test Accuracy**
        </td>
    </tr>
        <td>
        3-layer NN without regularization
        </td>
        <td>
        95%
        </td>
        <td>
        91.5%
        </td>
    <tr>
        <td>
        3-layer NN with L2-regularization
        </td>
        <td>
        94%
        </td>
        <td>
        93%
        </td>
    </tr>
    <tr>
        <td>
        3-layer NN with dropout
        </td>
        <td>
        93%
        </td>
        <td>
        95%
        </td>
    </tr>
</table> 
      
      
