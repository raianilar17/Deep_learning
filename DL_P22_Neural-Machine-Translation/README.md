# Neural-Machine-Translation

**I have built a Neural Machine Translation (NMT) model to translate human-readable dates ("25th of June, 2009") into machine-readable dates ("2009-06-25").**

**I have done this using an attention model, one of the most sophisticated sequence-to-sequence models.**

## Highlights of projects

- Machine translation models can be used to map from one sequence to another. They are useful not just for translating human languages (like French->English) but also for tasks like date format translation. 
- An attention mechanism allows a network to focus on the most relevant parts of the input when producing a specific part of the output. 
- A network using an attention mechanism can translate from inputs of length T_x to outputs of length T_y, where T_x and T_y can be different. 
- We can visualize attention weights to see what the network is paying attention to while generating each output.

## Translating human readable dates into machine readable dates

* The model I have built here could be used to translate from one language to another, such as translating from English to Hindi. 
* However, language translation requires massive datasets and usually takes days of training on GPUs. 
* To understand these models without using massive datasets, I have a simpler "date translation" project. 
* The network will input a date written in a variety of possible formats (*e.g. "the 29th of August 1958", "03/30/1968", "24 JUNE 1987"*) 
* The network will translate them into standardized, machine readable dates (*e.g. "1958-08-29", "1968-03-30", "1987-06-24"*). 
* We will have the network learn to output dates in the common machine-readable format YYYY-MM-DD. 

## Neural machine translation with attention

* If We had to translate a book's paragraph from French to English, We would not read the whole paragraph, then close the book and translate. 
* Even during the translation process, We would read/re-read and focus on the parts of the French paragraph corresponding to the parts of the English We are writing down. 
* The attention mechanism tells a Neural Machine Translation model where it should pay attention to at any step. 


###  Attention mechanism

In this part, I have implement the attention mechanism presented. 
* Here is a figure to remind us how the model works. 
    * The diagram on the left shows the attention model. 
    * The diagram on the right shows what one "attention" step does to calculate the attention variables $\alpha^{\langle t, t' \rangle}$.
    * The attention variables $\alpha^{\langle t, t' \rangle}$ are used to compute the context variable $context^{\langle t \rangle}$ for each timestep in the output ($t=1, \ldots, T_y$). 

<table>
<td> 
<img src="images/attn_model.png" style="width:500;height:500px;"> <br>
</td> 
<td> 
<img src="images/attn_mechanism.png" style="width:500;height:500px;"> <br>
</td> 
</table>
<caption><center> Figure 1: Neural machine translation with attention</center></caption>


## Files:

- [images](images) : Images 
- [models](models) : Pre-defined models
- [Neural_machine_translation_with_attention.ipynb](Neural_machine_translation_with_attention.ipynb) : complete project description
- [nmt_utils.py](nmt_utils.py) : pre-defined functions

## Commands to run code :
- Run jupyter notebook([Neural_machine_translation_with_attention.ipynb](Neural_machine_translation_with_attention.ipynb)) 
(The Jupyter Notebook is an open-source web application that allows us to create and share documents that contain live code, equations, 
visualizations and narrative text. Uses include: data cleaning and transformation, numerical simulation, statistical modeling, data visualization, 
machine learning, and much more.)
- Detail explanation is given in notebook. Please refer [Neural_machine_translation_with_attention.ipynb](Neural_machine_translation_with_attention.ipynb) .
- The notebook written in deep explanation with mathematical function.

## Future Work
- I will try to implement translating human languages (like French->English). 


