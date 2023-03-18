# Build_moon_classifier_with_different_gradients

## Optimization Methods :

Until now, I've always used Gradient Descent to update the parameters and minimize the cost. 
In this project, I will explain and used  more advanced optimization methods that can speed up learning and perhaps
even get me to a better final value for the cost function. Having a good optimization algorithm can be the difference between waiting days vs. 
just a few hours to get a good result. 

Gradient descent goes "downhill" on a cost function J. Think of it as trying to do this: 
<img src="images/cost.jpg" style="width:650px;height:300px;">
<caption><center> <u> Figure 1 </u>: Minimizing the cost is like finding the lowest point in a hilly landscape<br> At each step of the training,I update my parameters following a certain direction to try to get to the lowest possible point. </center></caption>


## Files :
- [datasets](datasets) : moon datasets
- [images](images): images related to project description
- [ICLR_conference_paper](ICLR.pdf): ADAM paper
- [Optimization_methods.ipynb](Optimization_methods.ipynb): complete project description
- [opt_utils_v1a.py](opt_utils_v1a.py): useful function
- [testCases.py](testCases.py): test case for verified function correction

## Commands to Run Code :
Run in jupyter notebook(The Jupyter Notebook is an open-source web application that allows you to create and share documents that contain live code, equations, visualizations and narrative text. Uses include: data cleaning and transformation, numerical simulation, statistical modeling, data visualization, machine learning, and much more.)

Detail explanation is given in notebook. [Please refer](Optimization_methods.ipynb).

The notebook written in deep explanation with mathematical function.

## Different Optimization Methods :

### 1. Gradient Descent (GD):

- A simple optimization method in machine learning is `gradient descent (GD)`. 
  When we take gradient steps with respect to all  training  examples on each step, it is also called `Batch Gradient Descent (BGD)`. 

### 2. Stochastic Gradient Descent (SGD) :
- SGD, which is equivalent to mini-batch gradient descent where each mini-batch has just 1 example. 

In Stochastic Gradient Descent, we use only 1 training example before updating the gradients. 
When the training set is large, SGD can be faster. But the parameters will "oscillate" toward the minimum rather than converge smoothly. 
Here is an illustration of this: 

<img src="images/kiank_sgd.png" style="width:750px;height:250px;">

 Figure 1   : SGD vs GD 
 
 "+" denotes a minimum of the cost.SGD leads to many oscillations to reach convergence. But each step is a lot faster to compute for SGD than for GD,
as it uses only one training example (vs. the whole batch for GD). 


### 3. Mini-batch gradient descent (MGD) :

In practice, we'll often get faster results if we do not use neither the whole training set,
nor only one training example, to perform each update. `Mini-batch gradient descent(MGD)` uses an intermediate number of examples for each step. 
With mini-batch gradient descent, we loop over the mini-batches instead of looping over individual training examples.

<img src="images/kiank_minibatch.png" style="width:750px;height:250px;">
<caption><center> <u> <font color='purple'> Figure 2 </u>: <font color='purple'> SGD vs Mini-Batch GD <br>

"+" denotes a minimum of the cost. Using mini-batches in our optimization algorithm often leads to faster optimization. </center></caption>

<font color='blue'>

#### Important Points:
- The difference between `gradient descent`, `mini-batch gradient descent` and `stochastic gradient descent` is the number of examples we use to perform one update step.
- we have to tune a learning rate hyperparameter alpha.
- With a well-turned mini-batch size, usually it outperforms either gradient descent or stochastic gradient descent (particularly when the training set is large).
  
### 4. Momentum :

- Because mini-batch gradient descent makes a parameter update after seeing just a subset of examples, 
  The direction of the update has some variance, and so the path taken by mini-batch gradient descent will "oscillate" toward convergence. 
  Using momentum can reduce these oscillations. 

- Momentum takes into account the past gradients to smooth out the update.
  We will store the 'direction' of the previous gradients in the variable v. 
  Formally, this will be the exponentially weighted average of the gradient on previous steps. 
  We can also think of v as the "velocity" of a ball rolling downhill, building up speed (and momentum) according to
  the direction of the gradient/slope of the hill. 

<img src="images/opt_momentum.png" style="width:400px;height:250px;">

Figure 3 : The red arrows shows the direction taken by one step of mini-batch gradient descent with momentum. 
The blue points show the direction of the gradient (with respect to the current mini-batch) on each step. 
Rather than just following the gradient, we let the gradient influence v and then take a step in the direction of v.

<font color='blue'>

#### Important Points to remember :

- Momentum takes past gradients into account to smooth out the steps of gradient descent.
  It can be applied with batch gradient descent, mini-batch gradient descent or stochastic gradient descent.
- We have to tune a momentum hyperparameter beta and a learning rate alpha.

### 5.RMSProp:
- The RMSprop optimizer is similar to the gradient descent algorithm with momentum. 
  The RMSprop optimizer restricts the oscillations in the vertical direction.
  Therefore, we can increase our learning rate and our algorithm could take larger steps in the horizontal direction converging faster. 
  The difference between RMSprop and gradient descent is on how the gradients are calculated. 

### 6. ADAM :

- Adam is one of the most effective optimization algorithms for training neural networks. It combines ideas from RMSProp and Momentum.

How does Adam work? :

- It calculates an exponentially weighted average of past gradients, and stores it in variables v (before bias correction) and  v_corrected  (with bias correction).
- It calculates an exponentially weighted average of the squares of the past gradients, and stores it in variables s (before bias correction) and  s_corrected (with bias correction).
- It updates parameters in a direction based on combining information from "1" and "2".

## Conclusions :

 - I built Model with different optimization algorithms with  "moons" dataset to test the different optimization methods. 
   (The dataset is named "moons" because the data from each of the two classes looks a bit like a crescent-shaped moon.)
   
 
 <table> 
    <tr>
        <td> 
        
  **optimization method**
        </td>
        <td>
        **accuracy**
        </td>
        <td>
        **cost shape**
        </td>
    </tr>
        <td>
        Gradient descent
        </td>
        <td>
        79.7%
        </td>
        <td>
        oscillations
        </td>
    <tr>
        <td>
        Momentum
        </td>
        <td>
        79.7%
        </td>
        <td>
        oscillations
        </td>
    </tr>
    <tr>
        <td>
        Adam
        </td>
        <td>
        94%
        </td>
        <td>
        smoother
        </td>
    </tr>
</table> 


Momentum usually helps, but given the small learning rate and the simplistic dataset, 
its impact is almost negligeable. Also, the huge oscillations I see in the cost come from the fact that some minibatches are more difficult than 
others for the optimization algorithm.

Adam on the other hand, clearly outperforms mini-batch gradient descent and Momentum.
When I run the model for more epochs on this simple dataset, all three methods will lead to very good results. 
However, We've seen that Adam converges a lot faster.

Some advantages of `Adam` include:

- Relatively low memory requirements (though higher than gradient descent and gradient descent with momentum) 
- Usually works well even with little tuning of hyperparameters (except alpha)

