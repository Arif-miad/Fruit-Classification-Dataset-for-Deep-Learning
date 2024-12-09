

---

# **Fruit Classification Dataset for Deep Learning**

## **About the Dataset**
This dataset contains images of 10 different fruits, collected and categorized into separate classes. It is designed for use in deep learning tasks such as training, testing, and prediction. The images are diverse and representative of common fruits found in supermarkets, making this dataset ideal for various fruit classification tasks.

### **Classes in the Dataset:**
- **Apple**
- **Orange**
- **Avocado**
- **Kiwi**
- **Mango**
- **Pineapple**
- **Strawberries**
- **Banana**
- **Cherry**
- **Watermelon**

### **Collection Methodology**
The dataset was collected through web scraping techniques from various sources such as Instagram and Google. The images were manually categorized into the respective classes mentioned above. This ensures a wide variety of fruit appearances, from different angles, lighting conditions, and backgrounds, making the dataset suitable for real-world applications.

### **Purpose of the Dataset**
This dataset can be used to:
- **Train** a deep learning model to classify fruits.
- **Test** the model's performance on unseen data.
- **Predict** the type of fruit based on new images.

### **Usage**
The dataset can be employed in multiple ways, including:
- **Image Classification**: Developing a deep learning model to classify fruits into their respective classes.
- **Data Augmentation**: Using techniques like rotation, flipping, and shifting to artificially increase the dataset size and improve model performance.
- **Transfer Learning**: Utilizing pre-trained models like MobileNetV2 for fine-tuning to suit fruit classification tasks.

### **Repository Structure**
- **/images**: Contains subfolders for each fruit class with images corresponding to the class.
- **/scraping_code**: Python scripts used to collect and organize the images.
- **README.md**: This file.
- **data_description.md**: Detailed descriptions of the dataset and its contents.

### **How to Use This Dataset**
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/fruit-classification-dataset.git
   cd fruit-classification-dataset
   ```

2. **Explore the dataset**:
   Navigate to the `/images` directory to see the class subfolders and the images within each.

3. **Preprocess the images**:
   - Rescale pixel values to [0, 1] for model training.
   - Apply data augmentation techniques like rotation, shifting, flipping, etc., using libraries such as `tensorflow.keras.preprocessing.image.ImageDataGenerator`.

4. **Train a model**:
   - Use this dataset to train a deep learning model.
   - Fine-tune models like MobileNetV2 or ResNet on this dataset for better performance.

5. **Test and predict**:
   - Split the dataset into training and validation sets.
   - Evaluate the model using the validation set.
   - Use the model for predictions on new images of fruits.

### **Contributing**
If you wish to contribute to this dataset:
- You can add more images of fruits.
- Modify the collection methodology.
- Improve the data augmentation techniques.
- Enhance the README file with additional insights or examples.

### **Contact**
- **Email**: arifmiahcse@gmail.com
- **LinkedIn**: [MY LinkedIn Profile](https://www.linkedin.com/in/arif-miah-8751bb217/)

---
This includes sections for importing libraries, loading the training and test images, visualizing the training images, working with specific fruit images (e.g., Apple and Banana), model building, transfer learning, and training the model. Each section is explained step by step:

---

## **Fruit Classification Project**

### **1. Import Libraries**
Before starting, we need to import all necessary libraries for the project. This includes libraries for data manipulation, image processing, deep learning model construction, and visualization.

```python
# Importing necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
```

### **2. Load The Train Images And Test Images**
Next, load the training and test images. The images are stored in directories for each class (e.g., `/images/apple`, `/images/banana`). I use `ImageDataGenerator` from Keras for augmenting and loading images in batches.

```python
# Paths to training and test images
train_dir = 'path/to/train/images'
test_dir = 'path/to/test/images'

# Data augmentation for training set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

# Data augmentation for test set
test_datagen = ImageDataGenerator(rescale=1./255)

# Loading training images
train_data_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Loading validation images
validation_data_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Loading test images
test_data_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # No need to shuffle test data
)
```

### **3. Visualizing The Train Images**
To understand the distribution and characteristics of the training images, visualize a few images from each category using Matplotlib.

```python
# Function to visualize a few images from each category
def plot_images(data_gen, columns=4, rows=2):
    plt.figure(figsize=(12, 8))
    for i, (images, labels) in enumerate(data_gen):
        for j in range(min(columns * rows, len(images))):
            plt.subplot(rows, columns, i * columns + j + 1)
            plt.imshow(images[j])
            plt.title(train_data_gen.class_indices)
            plt.axis('off')
        if i == rows * columns - 1:
            break
    plt.show()

plot_images(train_data_gen)
```

### **4. Apple Images**
Similarly, visualize specific fruit images like apples. This helps in understanding the variety within a class.

```python
# Function to visualize Apple images
def plot_class_images(data_gen, class_name, columns=4, rows=2):
    class_dir = os.path.join(train_dir, class_name)
    class_data_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
        class_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode=None
    )
    
    plt.figure(figsize=(12, 8))
    for i in range(min(columns * rows, len(class_data_gen))):
        plt.subplot(rows, columns, i + 1)
        image, _ = class_data_gen[i]
        plt.imshow(image[0])
        plt.title(class_name)
        plt.axis('off')
    plt.show()

plot_class_images(train_data_gen, 'Apple')
```

### **5. Banana Images**
Visualize images from another class, such as bananas, to see variations in the dataset.

```python
plot_class_images(train_data_gen, 'Banana')
```

### **6. Model Building**
Now, let's build a model using transfer learning. I use MobileNetV2, a popular pre-trained model suitable for image classification tasks.

```python
# Base model with pre-trained MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# Adding new layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dense(len(train_data_gen.class_indices), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=x)

# Summarize the model architecture
model.summary()
```

### **7. Transfer Learning**
Compile the model with a suitable optimizer and loss function. For this task, `Adam` optimizer and `categorical_crossentropy` are chosen.

```python
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
```

### **8. Check The Summary Of Model**
To review the structure and parameters of the model.

```python
model.summary()
```

### **9. Compile The Model**
Compile the model to prepare it for training.

```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

### **10. Model Training**
Train the model using the training and validation data.

```python
history = model.fit(
    train_data_gen,
    validation_data=validation_data_gen,
    epochs=10,
    steps_per_epoch=len(train_data_gen),
    validation_steps=len(validation_data_gen)
)
```

### **11. Plotting The Loss And Accuracy**
Visualize the model's performance with respect to training and validation accuracy and loss over epochs.

```python
# Plotting accuracy and loss curves
def plot_metrics(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(1, len(acc) + 1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.show()

plot_metrics(history)
```

### **12. Predictions**
Finally, evaluate the model on the test set and make predictions on new images.

```python
# Predict on test set
test_loss, test_acc = model.evaluate(test_data_gen)
print(f"Test accuracy: {test_acc}")

# Make predictions
predictions = model.predict(test_data_gen, batch_size=32)
```


### **Conclusion**
This guide walks you through the process of setting up, training, and evaluating a deep learning model for fruit classification using a transfer learning approach. The dataset, along with the augmentation and pre-processing techniques, prepares the model to handle real-world variations in fruit appearances effectively.





  
