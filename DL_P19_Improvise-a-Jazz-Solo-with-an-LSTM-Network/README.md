# Improvise-a-Jazz-Solo-with-an-LSTM-Network

In this project, I have implemented a model that uses an LSTM to generate music. I even be able to listen to my own music at the end of the project.

### Problem statement

Suppose, We would like to create a `jazz music` piece specially for a friend's birthday. 
However, We don't know any instruments or music composition. 
Fortunately, I know deep learning and have solved this problem using an LSTM network.  

I have trained a network to generate novel jazz solos in a style representative of a body of performed work.

<img src="images/jazz.jpg" style="width:450;height:300px;">

I have manipulate the preprocessing of the musical data to render it in terms of musical "values." 

### Details about music 
We can informally think of each "value" as a note, which comprises a pitch and duration. 
For example, if We press down a specific piano key for 0.5 seconds, then We have just played a note.
In music theory, a "value" is actually more complicated than this--specifically, it also captures the information needed to play multiple notes at the same time.
For example, when playing a music piece, We might press down two piano keys at the same time (playing multiple notes at the same time generates what's 
called a "chord"). But we don't need to worry about the details of music theory for this project. 

### Music as a sequence of values
* For the purpose of this project, all we need to know is that we will obtain a dataset of values, and will learn an RNN model to generate sequences of values. 
* Our music generation system will use 78 unique values. 

### Overview of our model

Here is the architecture of the model I used.

<img src="images/music_generation.png" style="width:600;height:400px;">

## Files:
- [Res_paper](Res_paper): useful papers regarding this projects
- [data](data): datasets
- [output](output): output of models
- [Improvise_a_Jazz_Solo_with_an_LSTM_Network.ipynb](Improvise_a_Jazz_Solo_with_an_LSTM_Network.ipynb): complete project code and details descriptions
- [data_utils.py](data_utils.py): pre-defined functions
- [grammar.py](grammar.py): pre-defined functions
- [inference_code.py](inference_code.py):pre-defined functions
- [midi.py](midi.py):pre-defined functions
- [music_utils.py](music_utils.py):pre-defined functions
- [preprocess.py](preprocess.py):pre-defined functions
- [qa.py](qa.py):pre-defined functions
- [requirements.txt](requirements.txt): packages required to run code
- [tune1.midi](tune1.midi): output of music generation

## Commands to run code :
- install all packages from [requirements.txt](requirements.txt) to run projects
- Run jupyter notebook([Improvise_a_Jazz_Solo_with_an_LSTM_Network.ipynb](Improvise_a_Jazz_Solo_with_an_LSTM_Network.ipynb)) 
  (The Jupyter Notebook is an open-source web application that allows us to create and share documents that contain live code, equations, 
  visualizations and narrative text. Uses include: data cleaning and transformation, numerical simulation, statistical modeling, data visualization, 
  machine learning, and much more.)
- Detail explanation is given in notebook. Please refer [Improvise_a_Jazz_Solo_with_an_LSTM_Network.ipynb](Improvise_a_Jazz_Solo_with_an_LSTM_Network.ipynb) .
- The notebook written in deep explanation with mathematical function.

## Key points:
- A sequence model can be used to generate musical values, which are then post-processed into midi music. 
- Fairly similar models can be used to generate dinosaur names , to generate music or to something else, with the major difference being the input fed to the model.  
- In Keras, sequence generation involves defining layers with shared weights, which are then repeated for the different time steps 1, ..., T_x. 
  
  
  ## Future Work:
  - I'll try to apply LSTM on different area(fields or projects).
  - In future , I'll try to improve accuracy of model and reduce loss function.







