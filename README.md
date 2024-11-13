Music Genre Classification with CNN
Project Overview
This project classifies the genre of a song using a Convolutional Neural Network (CNN). The model analyzes Mel Frequency Cepstral Coefficients (MFCCs) extracted from audio data to predict genres, leveraging a dataset of 1000 songs across 10 common genres. The model achieves an accuracy of around 70–90%, with further potential for improvement.

Dataset
Source: GTZAN Music Genre Dataset
Content: 1000 audio clips, each 30 seconds, covering 10 genres: blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae, and rock.
Preprocessing: Each song was divided into 10 segments, and MFCCs were extracted for each segment using the librosa library.
Model Architecture
Model Type: Convolutional Neural Network (CNN)
Layers:
Three 2D convolutional layers with ReLU activation
Max-pooling layers for feature reduction
Batch normalization and dropout for regularization
Optimizer: Adam with a learning rate of 0.0001
Loss Function: Categorical Cross-Entropy
Training: The model was trained for 30 epochs with a batch size of 32.
Results
The model achieved:

Training Accuracy: 75-95%
Testing Accuracy: 70-90%
The accuracy difference between training and testing data was minimal, indicating the model was not overfitted.

Single-Song Prediction
The model can also predict the genre of individual songs recorded or uploaded by the user. The MFCCs are extracted for each song segment, and the genre with the most frequent prediction among segments is selected.

How to Use
Clone this repository and install the required dependencies.
Prepare the dataset and process it with the MFCC feature extraction method.
Train the model on the dataset or load pre-trained weights.
Use the model to predict genres for new audio files.
Dependencies
TensorFlow and Keras for model implementation
Librosa for audio processing
Google Colab (optional) for GPU-accelerated training
Future Work
Increase model accuracy by training on a larger dataset.
Experiment with additional layers or different architectures like RNNs.
Deploy on GPU-based systems for faster training.
References
G. Tzanetakis and P. Cook, "Musical genre classification of audio signals," IEEE Transactions on Speech and Audio Processing, 2002.
GTZAN Dataset
Librosa Library Documentation
TensorFlow Documentation
