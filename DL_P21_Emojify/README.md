# Emojify!

In this project, I use word vector representations to build an Emojifier. 

Have you ever wanted to make your text messages more expressive? Your emojifier app will help you do that. 
So rather than writing:
>"Congratulations on the promotion! Let's get coffee and talk. Love you!"   

The emojifier can automatically turn this into:
>"Congratulations on the promotion! 👍 Let's get coffee and talk. ☕️ Love you! ❤️"

* I have implemented a model which inputs a sentence (such as "Let's go see the baseball game tonight!") and finds the most appropriate emoji to be used with this sentence (⚾️).

#### Using word vectors to improve emoji lookups
* In many emoji interfaces, We need to remember that ❤️ is the "heart" symbol rather than the "love" symbol. 
    * In other words, we'll have to remember to type "heart" to find the desired emoji, and typing "love" won't bring up that symbol.
* We can make a more flexible emoji interface by using word vectors!
* When using word vectors, we'll see that even if our training set explicitly relates only a few words to a particular emoji, our algorithm will be able to generalize and associate additional words in the test set to the same emoji.
    * This works even if those additional words don't even appear in the training set. 
    * This allows us to build an accurate classifier mapping from sentences to emojis, even using a small training set. 

#### What I built
1. In this project, I have started with a baseline model (Emojifier-V1) using word embeddings.
2. Then I built a more sophisticated model (Emojifier-V2) that further incorporates an LSTM. 

## 1 - Baseline model: Emojifier-V1

### 1.1 - Dataset EMOJISET

First built a simple baseline classifier. 

I have a tiny dataset (X, Y) where:
- X contains 127 sentences (strings).
- Y contains an integer label between 0 and 4 corresponding to an emoji for each sentence.

<img src="images/data_set.png" style="width:700px;height:300px;">
<caption><center> Figure 1: EMOJISET - a classification problem with 5 classes. A few examples of sentences are given here. </center></caption>

### 1.2 - Overview of the Emojifier-V1

In this segment, I have implemented a baseline model called "Emojifier-v1".  

<center>
<img src="images/image_1.png" style="width:900px;height:300px;">
<caption><center> Figure 2: Baseline model (Emojifier-V1).</center></caption>
</center>


#### Inputs and outputs
* The input of the model is a string corresponding to a sentence (e.g. "I love you). 
* The output will be a probability vector of shape (1,5), (there are 5 emojis to choose from).
* The (1,5) probability vector is passed to an argmax layer, which extracts the index of the emoji with the highest probability.


`
Training set:
Accuracy: 0.9772727272727273
Test set:
Accuracy: 0.8571428571428571
`

## Key points to remember from this section
- Even with a 127 training examples, we can get a reasonably good model for Emojifying. 
    - This is due to the generalization power word vectors gives us. 
- Emojify-V1 will perform poorly on sentences such as **"This movie is not good and not enjoyable"** 
    - It doesn't understand combinations of words.
    - It just averages all the words' embedding vectors together, without considering the ordering of words.
    
## 2 - Emojifier-V2: Using LSTMs in Keras: 

Built an LSTM model that takes word **sequences** as input!
* This model will be able to account for the word ordering. 
* Emojifier-V2 will continue to use pre-trained word embeddings to represent words.
* We will feed word embeddings into an LSTM.
* The LSTM will learn to predict the most appropriate emoji.

### 2.1 - Overview of the model

Here is the Emojifier-v2 I have implemented:

<img src="images/emojifier-v2.png" style="width:700px;height:400px;"> <br>
<caption><center> Figure 3: Emojifier-V2. A 2-layer LSTM sequence classifier. </center></caption>



`
Training set:
Accuracy: 0.9545
Test set:
Accuracy: 0.8214285629136222
`

## LSTM version accounts for word order
* Previously, Emojify-V1 model did not correctly label "not feeling happy," but our implementation of Emojiy-V2 got it right. 
    * (Keras' outputs are slightly random each time, so we may not have obtained the same result.) 
* The current model still isn't very robust at understanding negation (such as "not happy")
    * This is because the training set is small and doesn't have a lot of examples of negation. 
    * But if the training set were larger, the LSTM model would be much better than the Emojify-V1 model at understanding such complex sentences. 
    
## What we should remember
- If we have an NLP task where the training set is small, using word embeddings can help our algorithm significantly. 
- Word embeddings allow our model to work on words in the test set that may not even appear in the training set. 
- Training sequence models in Keras (and in most other deep learning frameworks) requires a few important details:
    - To use mini-batches, the sequences need to be **padded** so that all the examples in a mini-batch have the **same length**. 
    - An `Embedding()` layer can be initialized with pretrained values. 
        - These values can be either fixed or trained further on our dataset. 
        - If however our labeled dataset is small, it's usually not worth trying to train a large pre-trained set of embeddings.   
    - `LSTM()` has a flag called `return_sequences` to decide if you would like to return every hidden states or only the last one. 
    - We can use `Dropout()` right after `LSTM()` to regularize our network. 
    
 ## Files:
 
 - [data](data) : datasets
 - [images](images) : projects related images
 - [Emojify.ipynb](Emojify.ipynb) : complete project descriptions
 - [emo_utils.py](emo_utils.py) : Pre-defined function
 
## Commands to run code :

- Run jupyter notebook([Emojify.ipynb](Emojify.ipynb)) (The Jupyter Notebook is an open-source web application that allows us to create and share
documents that contain live code, equations, visualizations and narrative text. Uses include: data cleaning and transformation, numerical simulation, 
statistical modeling, data visualization, machine learning, and much more.)
- Detail explanation is given in notebook. Please refer [Emojify.ipynb](Emojify.ipynb) .
- The notebook written in deep explanation with mathematical function.

## Future Work
- I will choose the training set very large, So the LSTM model would be much better than the Emojify-V1 model at understanding such complex sentences. 



