import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

def main():
    # 1. Load and Preprocess Data
    print("Loading CIFAR-10 dataset...")
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    # 2. Visualize Data (Optional - usually for notebooks)
    # We'll skip showing the plot in a script unless requested, but here is the code:
    # plt.figure(figsize=(10,10))
    # for i in range(25):
    #     plt.subplot(5,5,i+1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(False)
    #     plt.imshow(train_images[i])
    #     plt.xlabel(class_names[train_labels[i][0]])
    # plt.show()

    # 3. Build the CNN Model
    print("Building the model...")
    model = models.Sequential()
    # Convolutional base
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    
    # Dense layers on top
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10)) # 10 output classes

    model.summary()

    # 4. Compile the Model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # 5. Train the Model
    print("Starting training...")
    history = model.fit(train_images, train_labels, epochs=10, 
                        validation_data=(test_images, test_labels))

    # 6. Evaluate the Model
    print("Evaluating model...")
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print(f'\nTest accuracy: {test_acc}')

    # 7. Save the model
    model.save('cifar10_model.h5')
    print("Model saved to cifar10_model.h5")

if __name__ == "__main__":
    main()
