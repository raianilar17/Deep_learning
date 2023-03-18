# Trigger-Word-Detection
I have constructed a speech dataset and implement an algorithm for trigger word detection (sometimes also called keyword detection, or wake word detection)

* Trigger word detection is the technology that allows devices like Amazon Alexa, Google Home, Apple Siri, and Baidu DuerOS to wake up upon hearing a certain word.  
* For this project,My trigger word will be "Activate." Every time it hears we say "activate," it will make a "chiming" sound. 
* By the end of this project,I able to record a clip of myself talking, and have the algorithm trigger a chime when it detects I said "activate." 
* After completing this project, perhaps We can also extend it to run on our laptop so that every time we say "activate" it starts up our favorite app, or turns on a network connected lamp in our house, or triggers some other event? 

<img src="images/sound.png" style="width:1000px;height:150px;">

In this project I learn to: 
- Structure a speech recognition project
- Synthesize and process audio recordings to create train/dev datasets
- Train a trigger word detection model and make predictions

## Highlights of projects:
- Data synthesis is an effective way to create a large training set for speech problems, specifically trigger word detection. 
- Using a spectrogram and optionally a 1D conv layer is a common pre-processing step prior to passing audio data to an RNN, GRU or LSTM.
- An end-to-end deep learning approach can be used to build a very effective trigger word detection system. 

# Files:
- [audio_examples](audio_examples) : audio examples
- [images](images) : images regarding projects
- [models](models) : pre-trained model
- [raw_data](raw_data) : raw data
- [Trigger_word_detection.ipynb](Trigger_word_detection.ipynb) : complete projects description
- [chime_output.wav](chime_output.wav) : sample audio
- [insert_test.wav](insert_test.wav) : sample audio
- [td_utils.py](td_utils.py) : pre-defined functions
- [train.wav](train.wav) :  train audio

# Commands to run code :
- Run jupyter notebook([Trigger_word_detection.ipynb](Trigger_word_detection.ipynb))
(The Jupyter Notebook is an open-source web application that allows us to create and share documents that contain live code, equations, 
visualizations and narrative text. Uses include: data cleaning and transformation, numerical simulation, statistical modeling, data visualization, 
machine learning, and much more.)
- Detail explanation is given in notebook. Please refer [Trigger_word_detection.ipynb](Trigger_word_detection.ipynb).
The notebook written in deep explanation with mathematical function.

# Future_work:
I will try to extend it to run on our laptop so that every time we say "activate" it starts up our favorite app, or turns on a network connected lamp in our house, or triggers some other event? 
