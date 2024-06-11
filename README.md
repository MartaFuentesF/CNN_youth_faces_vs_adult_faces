# Distinguishing Adult and Youth Faces Using Convolutional Neural Networks

## Introduction
The objective of this project is to develop a Convolutional Neural Network (CNN) model capable of distinguishing between the faces of adults and youth (up to the age of 13). Using TensorFlow and Keras, three CNN models were built and trained on a dataset comprising images of both adults and youth. This report provides an overview of the models, their performance, and potential applications.

## Potential Applications
While law enforcement is a primary potential user of this model, several other sectors could benefit from this technology:
1. **Healthcare Sector**: Early detection of developmental disorders and age-specific health conditions through analysis of facial features.
2. **Social Media Platforms**: Age verification to protect minors.
3. **Marketing and Retail**: Tailoring marketing strategies based on the age of customers.


## Methodology

### Data Acquisition


The data was acquired from Wernher (Vern) Krutein,the owner and creator of [photovault.com]('photovault.com')

Wernher Krutein is a digital artist, photographer, filmmaker, historian, and archivist. Over the past sixty years, he has built a vast archive that includes original negatives, slides, prints, videos, films, and various other media. His collection spans a wide range of subjects from Aerospace to Zimbabwe and includes contributions from over a hundred artists and photographers.

The images are meticulously cataloged, with each item labeled with relevant historical information. This extensive documentation adds significant value to the collection, making it a valuable resource for various applications.

I would like to express my sincere gratitude to Wernher Krutein for providing access to this invaluable resource. His lifelong dedication to archiving and preserving such a diverse range of images has made this project possible. For more information about his work and collection, please visit [photovault.com](https://photovault.com/). 

#### Image Dataset Information
The images for the project's data set are divided into the following:

| Image Category | Number of Items |
|----------------|-----------------|
| Images of Youth| 5,423           |
| Images of Adults| 9,212          |
| **Total Images**| **14,635**     |

## Image Examples

Below are some example images from the dataset:

<p align="center">
  <img src="https://github.com/MartaFuentesF/capstone_project/blob/main/images_and_graphs/image_examples/PLP_examples.JPG" alt="Samples of Portraits of Youth (PLP)" width="30%">
  <img src="https://github.com/MartaFuentesF/capstone_project/blob/main/images_and_graphs/image_examples/POR_examples.JPG" alt="Samples of Portraits of Adults (POR)" width="30%">
  <img src="https://github.com/MartaFuentesF/capstone_project/blob/main/images_and_graphs/image_examples/Problematic_images.JPG" alt="Samples of Potentially Problematic Images" width="30%">
</p>

The distinction between adults and youth was chosen at 13 years of age. Some overlap is present in the directories, which could lead to the model having difficulty classifying pre-teenager youth.

### Data Preparation

I prepared the data using two different approaches. First I will describe the approach that was successful. For interested readers, I will also describe the approach that was not successful, as an addendum at the bottom of this report.

The process I used is from ['a repository on github.com'], and it is copyrighted: Copyright 2018 The TensorFlow Authors.
In order to use keras' `image_dataset_from_directory()` function, a 'data_dir' directory was used; it containes the directories 'POR' (portraits of adults) and 'PLP' (portraits of youth). Using the process describe in the source cited, training and validation data sets were created as 'train_ds' and 'val_ds.' Each having a `validation_split` of 0.2. The result output is the following:

    - Found 14633 files belonging to 2 classes.
    - Using 11707 files for training.
    - Found 14633 files belonging to 2 classes.
    - Using 2926 files for validation.

After this, some EDA was conducted to see some of the images in the datasets. Below, you can see several sample images. Notice that while some images are traditional portraits, centered on single faces, many of the images aren't. This will likely affect how well my CNN is able to make distinctions. As a stretch goal, I hope to be able to weed through the images and remove the ones that are not of single faces.

As the reader may have noticed, there is a class imbalance in the classes of data the model will train on.

| Image Category | Number of Items |
|----------------|-----------------|
| Images of Youth| 5,423           |
| Images of Adults| 9,212          |
| **Total Images**| **14,635**     |


To address this, in CNNs two and three, class weights were used. With help from Argishti Ovsepyan and [stackoverflow.com]('https://stackoverflow.com/questions/66715975/class-weights-in-cnn'), class weights were applied to each of the classes. This is an alternative solution to augmenting the data via keras' [Image augmentation layers]('https://keras.io/api/layers/preprocessing_layers/image_augmentation/').

Finally, CNNs were built using different architectures. Each CNN was adjusted to make improvements on the previous attempt. Using the loss-function (binary-crossentropy) and accuracy as metrics to judge the success of the CNN in making correct predictions. 




### CNN Architectures
Two CNN models were built with different architectures:

#### CNN 1: Input Layer + 1 Hidden Layer

| Layer Type      | Filters | Kernel Size | Activation | Input Shape    | Additional Parameters |
|-----------------|---------|-------------|------------|----------------|-----------------------|
| Conv2D          | 512     | 2           | relu       | (256, 256, 3)  |                       |
| MaxPooling2D    |         | 2           |            |                |                       |
| Conv2D          | 64      | 2           | relu       |                |                       |
| MaxPooling2D    |         | 2           |            |                |                       |
| Flatten         |         |             |            |                |                       |
| Dense           |         |             | sigmoid    |                | Output: 1 neuron      |

#### CNN 2: Input Layer + 2 Hidden Layers

| Layer Type      | Filters | Kernel Size | Activation | Input Shape    | Additional Parameters |
|-----------------|---------|-------------|------------|----------------|-----------------------|
| Conv2D          | 512     | 3           | relu       | (256, 256, 3)  |                       |
| MaxPooling2D    |         | 2           |            |                | padding='same'        |
| Conv2D          | 256     | 3           | relu       |                |                       |
| MaxPooling2D    |         | 2           |            |                | padding='same'        |
| Conv2D          | 256     | 3           | relu       |                |                       |
| MaxPooling2D    |         | 2           |            |                | padding='same'        |
| Flatten         |         |             |            |                |                       |
| Dense           |         |             | sigmoid    |                | Output: 1 neuron      |

Note: A choice was made to use one output neuron, with sigmoid activation, as recommended by 'itdxer' in the following [Stack Exchange]('https://stats.stackexchange.com/questions/207049/neural-network-for-binary-classification-use-1-or-2-output-neurons').

### Performance Metrics
The performance of the models was evaluated using accuracy and loss metrics:

| Model | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy |
|-------|---------------|-------------------|-----------------|---------------------|
| CNN 1 | 0.4773        | 0.7526            | 0.5494          | 0.7180              |
| CNN 2 | 0.3744        | 0.8161            | 0.6049          | 0.7397              |


## Results and Observations

### Model Performance
CNN 2 outperformed CNN 1 in terms of training accuracy but showed signs of overfitting as indicated by a higher validation loss and a slightly lower validation accuracy. This suggests that while CNN 2 is more complex and captures more features, it may also be less generalizable to unseen data.


### Graphs
Below are the graphs of binary-crossentropy versus epochs and accuracy as a function of epochs for both models.

**Training and Validation Loss vs. Epochs**

*Insert Graphs Here*

**Training and Validation Accuracy vs. Epochs**

*Insert Graphs Here*

## Technical Insights

### Convolutional Layers
The number of filters and the kernel size in convolutional layers impact the model's ability to detect features. Larger filters can capture more complex patterns, while smaller kernel sizes focus on finer details.

### Pooling Layers
MaxPooling layers help reduce the spatial dimensions of the feature maps, reducing the computational load and the risk of overfitting. Padding in pooling layers ensures that the spatial dimensions are preserved, which can be crucial for deeper networks.

### Regularization
Regularization techniques such as Dropout and L2 regularization help prevent overfitting by introducing noise during training and penalizing large weights, respectively.

### Loss Function and Optimizer
Binary_crossentropy was chosen as the loss function because the problem is a binary classification. The Adam optimizer is used for its efficiency and adaptability, providing a good balance between speed and performance.

## Challenges and Learnings

### Computational Resources
The models required substantial computational power and time to fit. Each model took five to ten hours to fit, using an M3 chip with 18GB of memory. Future work could explore using GPU acceleration or cloud-based solutions to optimize the training process.

### Understanding Layers
Learning about the various layers in CNNs and their roles was instrumental in designing effective models. Convolutional layers detect features, pooling layers reduce dimensions, and dense layers make final predictions.

## Future Work
Future improvements could include experimenting with more advanced architectures like ResNet, implementing data augmentation, and exploring other regularization techniques. Additionally, evaluating the model using other metrics such as precision, recall, and F1-score could provide more insights into its performance.

## Conclusion
This project successfully developed and evaluated two CNN models for distinguishing between faces of adults and youth. While the models showed promising results, further improvements and optimizations are necessary for real-world applications. The potential applications of this model span various sectors, highlighting its versatility and impact.

## References
*List of references and sources used for guidance.*



---
**Addendum: The unsuccessful approach:**
Initially, the data set was organized into a specific structure,  then split using the 'validation_split' argument to sub for training and validation data. 

sklearn's `train_test_split()`. 
This became the structure of my data:

## INSERT IMAGE

I then used keras' `image_dataset_from_directory()` function to load the images for my model to use. Unfortunately, this approach was not useful, as it augmented the data by roughly 8,500 images, and finding the bug was doing that was not possible within the time parameters for this project. 

---

