# Flower Species Identifier
## Project Description
This project is a machine learning model designed to classify flower species based on image data. The model leverages a Convolutional Neural Network (CNN) to learn and predict the species of flowers from a dataset of images. The project aims to achieve high accuracy in identifying different flower species through image processing and deep learning techniques.

## Project Structure
The project is structured as a Jupyter notebook that includes data preprocessing, model training, and evaluation. The key steps involved in the project are:

### 1. Data Loading and Preprocessing:
      Loading the dataset of flower images.
      Splitting the data into training and testing sets.
      Normalizing and augmenting the image data to improve model performance.
      
### 2. Model Building:
      Defining a Convolutional Neural Network (CNN) architecture.
      Compiling the model with appropriate loss functions and optimizers.
      Training the model on the training dataset.
      
### 3. Model Evaluation:
      Evaluating the trained model on the test dataset.
      Measuring the accuracy and loss to assess the model's performance.

## Dataset
The dataset used in this project contains images of flowers belonging to five different species. The images are preprocessed to standardize their size and scale before being fed into the model.

## Model Architecture
The model architecture includes the following layers:

Convolutional Layers: For feature extraction from the images.
Max Pooling Layers: To reduce the dimensionality of the feature maps.
Dropout Layer: To prevent overfitting.
Flatten Layer: To convert the 2D feature maps into a 1D feature vector.
Dense Layers: To perform the classification based on the extracted features.

## Training
The model is trained for 50 epochs using the Adam optimizer and Sparse Categorical Crossentropy loss function. Data augmentation techniques are applied during training to enhance the model's ability to generalize to new images.

## Evaluation
The model's performance is evaluated using accuracy and loss metrics. The final trained model achieves an accuracy of approximately 96.5% on the training data and 75.7% on the test data.

## Results
The results demonstrate the effectiveness of the CNN model in identifying flower species from images, with high training accuracy and reasonable test accuracy.

