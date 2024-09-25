# Pytorch LSTM Music Generator
Music generator using Pytorch LSTM

# Requirements
Python 3.x

Torch and music21 

GPU strongly recommended for training

# Info
This project is inspired by Sigurður Skúli's towards data science article ['How to Generate Music using a LSTM Neural Network in Keras'](https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5), and the subsequently work of jordan-bird on (https://github.com/jordan-bird/Keras-LSTM-Music-Generator.git)

The objective of this project was to implement something similar using Pytorch instead. We use the LSTM model to predict the Notes, Chords, the offsets of the note from the previous one and the durations of the notes in the sequence.

# Use 
Everything in this repo can be run as-is to train on video game piano pieces and generate music:

1. Run train.py for a while until note/chord loss is below 1
2. Change line 53 in train.py to specify where your .midi files are stored
3. Set line 93 in generate-music.py to the name of the .pth file with the trained model
4. Run generate-music.py to generate a midi file from the model

