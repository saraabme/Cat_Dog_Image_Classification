# Cat and Dog Image Classifier

<img src ='CatDog.jpg'></img>

## Introduction
This project utilizes deep learning techniques, specifically convolutional neural networks (ConvNets), to distinguish between images of cats and dogs. Built using PyTorch, a leading deep learning library, this classifier demonstrates the application of ConvNets to binary image classification tasks in computer vision.

The classifier is developed to showcase practical implementations of neural networks in interpreting visual data. It uses the "Dogs vs. Cats" dataset from Kaggle to train and validate the model, employing image preprocessing techniques, batch normalization, ReLU activations, and optional integration of pretrained models like ResNet to enhance accuracy and performance.

Designed to operate on both CPU and GPU environments, the project ensures functionality across various hardware configurations. The architecture supports easy experimentation with different hyperparameters and model settings, making it straightforward to adapt and optimize according to specific requirements.

The model’s structure and its training process are clearly documented, facilitating a deeper understanding of neural network operations, training procedures, and performance evaluation. This project not only serves as a robust tool for practical machine learning applications but also as a detailed example for those exploring advanced image classification systems using deep learning technologies.





## Project Structure
- `model.py`: Contains the definition of the ConvNet and the neural network blocks used.
- `load.py`: (In the data folder) Includes functions for data loading and transformations.
- `train.py`: Script for training the model with options for hyperparameters.


### Installation
To run this project, you need Python 3.8 or later, PyTorch, and torchvision. You can install the required libraries using pip:

```bash
pip install torch torchvision
```

### Dataset
The dataset used is a standard dogs vs. cats dataset which can be downloaded from popular datasets repositories or Kaggle. Adjust the path in the data.py script to where you have stored the dataset.

An example of where to download the dataset is [here](https://www.kaggle.com/datasets/tongpython/cat-and-dog), [here](https://www.kaggle.com/c/dogs-vs-cats/data).


### Training the Model
To train the model, run the `train.py` script. You can specify the hyperparameters such as the learning rate, batch size, and number of epochs.
```
python train.py --batch_size 32 --n_epochs 50 --use_cuda
```

### Evaluating the Model
After training, you can evaluate the model on a validation dataset to check its performance.


## Configuration
You can adjust various parameters and settings using command line arguments in the training script:

--batch_size: The size of each batch during training.
--n_epochs: Number of training epochs.
--use_cuda: Flag to enable CUDA (GPU acceleration), if available.


## Model Details
The model architecture is defined in model.py. It uses a series of convolutional layers and optionally integrates pretrained models for enhanced performance. The model employs batch normalization and ReLU activations to improve convergence.
