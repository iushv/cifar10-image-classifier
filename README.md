# CIFAR-10 Image Classification with CNN

A comprehensive deep learning project for classifying images from the CIFAR-10 dataset using Convolutional Neural Networks (CNN) built with TensorFlow/Keras.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Visualizations](#visualizations)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset. The model is trained to recognize 10 different classes of objects with high accuracy using modern deep learning techniques.

**Key Highlights:**
- ✅ Advanced CNN architecture with batch normalization
- ✅ Comprehensive data visualization and analysis
- ✅ Training with callbacks (early stopping, learning rate reduction)
- ✅ Detailed performance metrics and confusion matrix
- ✅ Ready-to-use Jupyter notebook for Google Colab
- ✅ Standalone Python script for production use

## 📊 Dataset

**CIFAR-10** is a well-known dataset in computer vision and machine learning:

- **Total Images:** 60,000 color images (32×32 pixels)
- **Training Set:** 50,000 images
- **Test Set:** 10,000 images
- **Classes:** 10 categories

### Classes:
1. ✈️ Airplane
2. 🚗 Automobile
3. 🐦 Bird
4. 🐱 Cat
5. 🦌 Deer
6. 🐕 Dog
7. 🐸 Frog
8. 🐴 Horse
9. 🚢 Ship
10. 🚚 Truck

## ✨ Features

### Data Processing
- Automatic dataset download and loading
- Image normalization (pixel values scaled to 0-1)
- Data visualization with matplotlib

### Model Features
- **Convolutional Layers:** 3 blocks with increasing filters (32 → 64 → 128)
- **Batch Normalization:** For faster convergence and stability
- **Dropout Regularization:** To prevent overfitting (25% and 50%)
- **MaxPooling:** For spatial dimension reduction
- **Dense Layers:** Fully connected layers for classification

### Training Optimizations
- **Early Stopping:** Prevents overfitting by monitoring validation loss
- **Learning Rate Reduction:** Adaptive learning rate scheduling
- **Adam Optimizer:** Efficient gradient descent optimization

### Evaluation & Analysis
- Training/validation accuracy and loss plots
- Confusion matrix visualization
- Per-class accuracy breakdown
- Classification report with precision, recall, F1-score
- Sample predictions with confidence scores

## 📁 Project Structure

```
Image_classification/
│
├── cifar10_image_classifier.ipynb    # Main Jupyter notebook (recommended)
├── cifar10_classifier.py              # Standalone Python script
├── README.md                          # Project documentation
│
├── cifar10_cnn_model.h5              # Saved model (after training)
├── cifar10_cnn_model/                # SavedModel format (after training)
│
└── image.ipynb                        # Additional notebook
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone or Download the Project

```bash
cd Image_classification
```

### Step 2: Install Required Libraries

```bash
pip install tensorflow numpy matplotlib scikit-learn seaborn jupyter
```

Or use a requirements file:

```bash
pip install -r requirements.txt
```

### Requirements.txt
```
tensorflow>=2.10.0
numpy>=1.23.0
matplotlib>=3.5.0
scikit-learn>=1.1.0
seaborn>=0.12.0
jupyter>=1.0.0
```

## 💻 Usage

### Option 1: Jupyter Notebook (Recommended)

#### Local Execution:
```bash
jupyter notebook cifar10_image_classifier.ipynb
```

#### Google Colab:
1. Upload `cifar10_image_classifier.ipynb` to Google Colab
2. Click **Runtime → Run all**
3. The notebook will automatically download the dataset and train the model

### Option 2: Python Script

```bash
python cifar10_classifier.py
```

This will:
1. Load the CIFAR-10 dataset
2. Build and compile the CNN model
3. Train for 10 epochs
4. Evaluate on the test set
5. Save the model as `cifar10_model.h5`

## 🏗️ Model Architecture

```
Model: Sequential
_________________________________________________________________
Layer (type)                 Output Shape              Params
=================================================================
Conv2D                       (None, 32, 32, 32)        896
BatchNormalization           (None, 32, 32, 32)        128
Conv2D                       (None, 32, 32, 32)        9,248
BatchNormalization           (None, 32, 32, 32)        128
MaxPooling2D                 (None, 16, 16, 32)        0
Dropout                      (None, 16, 16, 32)        0
_________________________________________________________________
Conv2D                       (None, 16, 16, 64)        18,496
BatchNormalization           (None, 16, 16, 64)        256
Conv2D                       (None, 16, 16, 64)        36,928
BatchNormalization           (None, 16, 16, 64)        256
MaxPooling2D                 (None, 8, 8, 64)          0
Dropout                      (None, 8, 8, 64)          0
_________________________________________________________________
Conv2D                       (None, 8, 8, 128)         73,856
BatchNormalization           (None, 8, 8, 128)         512
Conv2D                       (None, 8, 8, 128)         147,584
BatchNormalization           (None, 8, 8, 128)         512
MaxPooling2D                 (None, 4, 4, 128)         0
Dropout                      (None, 4, 4, 128)         0
_________________________________________________________________
Flatten                      (None, 2048)              0
Dense                        (None, 128)               262,272
BatchNormalization           (None, 128)               512
Dropout                      (None, 128)               0
Dense                        (None, 10)                1,290
=================================================================
Total params: 552,874
Trainable params: 551,722
Non-trainable params: 1,152
```

### Hyperparameters:
- **Optimizer:** Adam
- **Loss Function:** Sparse Categorical Crossentropy
- **Batch Size:** 64
- **Epochs:** 50 (with early stopping)
- **Learning Rate:** Adaptive (reduced on plateau)

## 📈 Results

### Expected Performance:
- **Test Accuracy:** ~75-85%
- **Training Time:** ~10-15 minutes (with GPU)
- **Model Size:** ~6.5 MB

### Performance Metrics:
The model provides detailed metrics including:
- Overall accuracy and loss
- Per-class precision, recall, and F1-score
- Confusion matrix
- Training/validation curves

## 📊 Visualizations

The notebook includes several visualizations:

1. **Sample Images:** Display of 25 random training images
2. **Training History:** Accuracy and loss curves over epochs
3. **Predictions:** Visual comparison of predicted vs. actual labels
4. **Confusion Matrix:** Heatmap showing classification performance
5. **Per-Class Accuracy:** Bar chart for each class

## 🔧 Customization

### Modify Model Architecture:
Edit the `create_model()` function in the notebook to:
- Add more convolutional layers
- Change filter sizes
- Adjust dropout rates
- Modify dense layer sizes

### Adjust Training Parameters:
```python
history = model.fit(
    train_images, 
    train_labels,
    epochs=100,        # Change number of epochs
    batch_size=32,     # Modify batch size
    validation_split=0.2  # Use validation split instead of test set
)
```

### Data Augmentation:
Add data augmentation for better generalization:
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
```

## 🎓 Learning Resources

- [TensorFlow Documentation](https://www.tensorflow.org/tutorials)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [CNN Architectures](https://cs231n.github.io/convolutional-networks/)
- [Deep Learning Book](https://www.deeplearningbook.org/)

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- CIFAR-10 dataset by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton
- TensorFlow and Keras teams for the excellent framework
- The deep learning community for tutorials and resources

## 📧 Contact

For questions or feedback, please open an issue in the repository.

---

**Happy Learning! 🚀**

*Built with ❤️ using TensorFlow and Keras*
