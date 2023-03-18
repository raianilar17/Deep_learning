# Face Recognition system

In this Project, I built a face recognition system. Many of the ideas presented here are from [FaceNet](https://arxiv.org/pdf/1503.03832.pdf). 
and
[DeepFace](https://research.fb.com/wp-content/uploads/2016/11/deepface-closing-the-gap-to-human-level-performance-in-face-verification.pdf). 

Face recognition problems commonly fall into two categories: 

- **Face Verification** - "is this the claimed person?". For example, at some airports, you can pass through customs by letting a system scan your passport and then verifying that you (the person carrying the passport) are the correct person. A mobile phone that unlocks using your face is also using face verification. This is a 1:1 matching problem. 
- **Face Recognition** - "who is this person?". For example, the video lecture showed a [face recognition video](https://www.youtube.com/watch?v=wr4rx0Spihs) of Baidu employees entering the office without needing to otherwise identify themselves. This is a 1:K matching problem. 

FaceNet learns a neural network that encodes a face image into a vector of 128 numbers. By comparing two such vectors, I can then determine if two pictures are of the same person.
    
    
## Face Recognition Applications :

**Face recognition is in human computer interaction,
virtual reality, database recovery, multimedia, computer
entertainment, information security e.g. operating system,
medical records, online banking., Biometric e.g. Personal
Identification - Passports, driver licenses , Automated identity
verification - border controls , Law enforcement e.g. video
surveillances , investigation , Personal Security - driver
monitoring system, home video surveillance system.**    
    
        
## Highlights of  project :

- Implement the triplet loss function
- Use a pretrained model to map face images into 128-dimensional encodings
- Use these encodings to perform face verification and face recognition

## Files :

- [Res_paper](Res_paper) : face reccognition related paper
- [datasets](datasets)   : datasets
- [images](images)       : images for testing purpose
- [weights](weights)     : pre-trained model weights
- [Face_Recognition.ipynb](Face_Recognition.ipynb): complete project description
- [fr_utils.py](fr_utils.py): some useful functions
- [inception_blocks_v2.py](inception_blocks_v2.py): model
- [nn4.small2.v7.h5](nn4.small2.v7.h5): pre-trained weights

## Commands to Run Code :

- Run jupyter notebook([Face_Recognition.ipynb](Face_Recognition.ipynb)) (The Jupyter Notebook is an open-source web application that allows you to create and share documents that contain live code, equations, visualizations and narrative text. Uses include: data cleaning and transformation, numerical simulation, statistical modeling, data visualization, machine learning, and much more.)
- Detail explanation is given in notebook. [Please refer](Face_Recognition.ipynb) .
- The notebook written in deep explanation with mathematical function.

The code succesfully run on Keras==2.0.3 and tensorflow==1.15.0.

## Key points to remember :

- Face verification solves an easier 1:1 matching problem; face recognition addresses a harder 1:K matching problem. 
- The triplet loss is an effective loss function for training a neural network to learn an encoding of a face image.
- The same encoding can be used for verification and recognition. 
  Measuring distances between two images' encodings allows you to determine whether they are pictures of the same person. 

## Future Work :

Although I won't implement it here, here are some ways to further improve the algorithm:
- Put more images of each person (under different lighting conditions, taken on different days, etc.) into the database. 
Then given a new image, compare the new face to multiple pictures of the person. This would increase accuracy.
- Crop the images to just contain the face, and less of the "border" region around the face. 
This preprocessing removes some of the irrelevant pixels around the face, and also makes the algorithm more robust.
