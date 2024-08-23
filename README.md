# Cat_Dog-Classifier using CNN

This project implements a Convolutional Neural Network (CNN) for binary image classification to differentiate between images of dogs and cats. The model, built using TensorFlow and Keras, achieves over 75% accuracy on the validation set.

Model Architecture

Convolutional Layers: Three Conv2D layers with ReLU activation are used to extract features from the images. ReLU (Rectified Linear Unit) introduces non-linearity, helping the model to learn complex patterns.

Pooling Layers: MaxPooling layers follow each convolutional layer to reduce the spatial dimensions, making the model computationally efficient.

Batch Normalization: Batch Normalization layers are added after each convolutional layer to stabilize and accelerate training.

Dense Layers: Two dense layers with ReLU activation are added before the final output layer to further refine the features.

Dropout: Dropout layers are used to prevent overfitting by randomly setting a fraction of input units to 0 during training.

Output Layer: A dense layer with a sigmoid activation function provides the final classification output (dog or cat).

Visualizations

The project includes plots of the training and validation accuracy and loss over 10 epochs, showing the model's performance.

Sample Prediction

A sample prediction on a test image demonstrates the model's ability to classify an unseen image as either a dog or a cat.
