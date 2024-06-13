# Distinguishing Adult and Youth Faces Using Convolutional Neural Networks

## Introduction
The objective of this project is to develop a Convolutional Neural Network (CNN) model capable of distinguishing between the faces of adults and youth (up to the age of 13). Using TensorFlow and Keras, five CNN models were built and trained on a dataset comprising images of both adults and youth. This report provides an overview of the models, their performance, and potential applications. The focus of this project was on improving the performance of CNNs by adding regularizers and adjusting arguments in the `.fit()` and `.compile()` methods. To this end, five CNNs were built and evaluated using the value of their loss functions (binary-crossentropy), accuracy as well as an F1 score for the fourth model.

### Important Considerations
* These models require significant computing power. Each took approximately 10 hours to fit using an M3 chip with 18GB of memory.
* Consider creating a separate Keras environment.
* Consider working in Google Colab for better computational resources.

## Potential Applications
While law enforcement is a primary potential user of this model, several other sectors could benefit from this technology:
1. **Healthcare Sector**: Early detection of developmental disorders and age-specific health conditions through analysis of facial features.
2. **Social Media Platforms**: Age verification to protect minors.
3. **Marketing and Retail**: Tailoring marketing strategies based on the age of customers.

## Methodology

### Data Acquisition
The data was acquired from Wernher (Vern) Krutein, the owner and creator of [photovault.com](https://photovault.com).

Wernher Krutein is a digital artist, photographer, filmmaker, historian, and archivist. Over the past sixty years, he has built a vast archive that includes original negatives, slides, prints, videos, films, and various other media. His collection spans a wide range of subjects from Aerospace to Zimbabwe and includes contributions from over a hundred artists and photographers.

The images are meticulously cataloged, with each item labeled with relevant historical information. This extensive documentation adds significant value to the collection, making it a valuable resource for various applications.

After contacting Vern, he generously agreed to provide access to the images. I would like to express my sincere gratitude to Wernher Krutein for providing access to this invaluable resource. His lifelong dedication to archiving and preserving such a diverse range of images has made this project possible. For more information about his work and collection, please visit [photovault.com](https://photovault.com).

#### Image Dataset Information

The images are 800 x 1200 pixel JPEG files.

| Image Category   | Number of Items |
|------------------|-----------------|
| Images of Youth  | 5,423           |
| Images of Adults | 9,212           |
| **Total Images** | **14,635**      |

The distinction between adults and youth was chosen at 13 years of age. Some overlap is present in the directories, which could lead to the model having difficulty classifying pre-teenager youth.

### Data Preparation

Data preparation was done using two approaches. The successful approach is outlined below, with a description of the unsuccessful attempt provided as an addendum for those interested in the detailed process.

The process used is from [a repository on github]('https://github.com/tensorflow/docs/blob/master/site/en/tutorials/images/classification.ipynb') and is copyrighted: Copyright 2018 The TensorFlow Authors. In order to use Keras' `image_dataset_from_directory()` function, a 'data_dir' directory was created; it contains the directories 'POR' (portraits of adults) and 'PLP' (portraits of youth). Using the process described in the cited source, training and validation datasets were created as 'train_ds' and 'val_ds,' each having a `validation_split` of 0.2. The result output is as follows:

    - Found 14633 files belonging to 2 classes.
    - Using 11707 files for training.
    - Found 14633 files belonging to 2 classes.
    - Using 2926 files for validation.

After loading the images, I performed some exploratory data analysis (EDA) to visualize the dataset. The samples below highlight a mix of image types: some are traditional portraits centered on individual faces, while others are not. Some of the images in both classes are of several people together or full-body images. This variety might influence how well the CNN learns to distinguish between adult and youth faces. As a future improvement, I aim to filter out images that do not feature single, centered faces to potentially improve the model's performance.

### Image Examples

Below are some example images from the datasets:

<p align="center">
  <img src="https://github.com/MartaFuentesF/capstone_project/blob/main/images_and_graphs/image_examples/PLP_examples.JPG" alt="Samples of Portraits of Youth (PLP)" width="30%">
  <img src="https://github.com/MartaFuentesF/capstone_project/blob/main/images_and_graphs/image_examples/POR_examples.JPG" alt="Samples of Portraits of Adults (POR)" width="30%">
  <img src="https://github.com/MartaFuentesF/capstone_project/blob/main/images_and_graphs/image_examples/Problematic_images.JPG" alt="Samples of Potentially Problematic Images" width="30%">
</p>


As the reader may have noticed, there is a class imbalance in the data the model will train on.

| Image Category   | Number of Items |
|------------------|-----------------|
| Images of Youth  | 5,423           |
| Images of Adults | 9,212           |
| **Total Images** | **14,635**      |

To address this, class weights were used in CNNs two, three, and four, along with other regularization techniques and parameter adjustments. 

With the help of Argishti Ovsepyan and resources from [Stack Overflow](https://stackoverflow.com/questions/66715975/class-weights-in-cnn), class weights were applied to balance the classes. This is an alternative to augmenting the data via Keras' [image augmentation layers](https://keras.io/api/layers/preprocessing_layers/image_augmentation/).

Different architectures were employed for each CNN to iteratively improve performance. The models were evaluated using the binary cross-entropy loss function and accuracy as primary metrics, with additional evaluation using the F1 score and the Area Under the Curve (AUC) for the Receiver Operating Characteristic (ROC) curve.

Note: A single output neuron with sigmoid activation was chosen for this binary classification task, as recommended by 'itdxer' on [Stack Exchange](https://stats.stackexchange.com/questions/207049/neural-network-for-binary-classification-use-1-or-2-output-neurons).

The CNN will predict the probabilities for the validation set. The predicted probabilities are thresholded at 0.5 to convert them to class labels (0 or 1). Boolean values are returned. More on this in notebook cnn4.



### CNN Architectures
Four CNN models were built with different architectures:

#### CNN 1: Input Layer + 1 Hidden Layer

| Layer Type      | Filters | Kernel Size | Activation | Input Shape    | Additional Parameters                       |
|-----------------|---------|-------------|------------|----------------|---------------------------------------------|
| Conv2D          | 512     | 2           | relu       | (256, 256, 3)  |                                             |
| MaxPooling2D    |         | 2           |            |                |                                             |
| Conv2D          | 64      | 2           | relu       |                |                                             |
| MaxPooling2D    |         | 2           |            |                |                                             |
| Flatten         |         |             |            |                |                                             |
| Dense           |         |             | sigmoid    |                | Output: 1 neuron                            |

#### CNN 2: Input Layer + 2 Hidden Layers with Adjusted Filters and Kernel Size

| Layer Type      | Filters | Kernel Size | Activation | Input Shape    | Additional Parameters                       |
|-----------------|---------|-------------|------------|----------------|---------------------------------------------|
| Conv2D          | 512     | 3           | relu       | (256, 256, 3)  |                                             |
| MaxPooling2D    |         | 2           |            |                | padding='same'                              |
| Conv2D          | 256     | 3           | relu       |                | kernel_regularizer=regularizers.l2(0.01)    |
| MaxPooling2D    |         | 2           |            |                | padding='same'                              |
| Conv2D          | 256     | 3           | relu       |                | kernel_regularizer=regularizers.l2(0.01)    |
| MaxPooling2D    |         | 2           |            |                | padding='same'                              |
| Flatten         |         |             |            |                |                                             |
| Dense           |         |             | sigmoid    |                | Output: 1 neuron                            |

#### CNN 3: Input Layer + Multiple Hidden Layers with Dropout

| Layer Type      | Filters | Kernel Size | Activation | Input Shape    | Additional Parameters                       |
|-----------------|---------|-------------|------------|----------------|---------------------------------------------|
| Conv2D          | 512     | 3           | relu       | (256, 256, 3)  |                                             |
| MaxPooling2D    |         | 2           |            |                | padding='same'                              |
| Dropout         |         |             |            |                | rate=0.5                                    |
| Conv2D          | 256     | 3           | relu       |                | kernel_regularizer=regularizers.l2(0.01)    |
| MaxPooling2D    |         | 2           |            |                | padding='same'                              |
| Dropout         |         |             |            |                | rate=0.5                                    |
| Conv2D          | 256     | 3           | relu       |                | kernel_regularizer=regularizers.l2(0.01)    |
| MaxPooling2D    |         | 2           |            |                | padding='same'                              |
| Dropout         |         |             |            |                | rate=0.5                                    |
| Flatten         |         |             |            |                |                                             |
| Dense           |         |             | sigmoid    |                | Output: 1 neuron                            |


#### CNN 4: Input Layer + Multiple Hidden Layers with Dropout

| Layer Type      | Filters | Kernel Size | Activation | Input Shape    | Additional Parameters                       |
|-----------------|---------|-------------|------------|----------------|---------------------------------------------|
| Conv2D          | 512     | 3           | relu       | (256, 256, 3)  |                                             |
| MaxPooling2D    |         | 2           |            |                | padding='same'                              |
| Dropout         |         |             |            |                | rate=0.5                                    |
| Conv2D          | 256     | 3           | relu       |                | kernel_regularizer=regularizers.l2(0.03)    |
| MaxPooling2D    |         | 2           |            |                | padding='same'                              |
| Dropout         |         |             |            |                | rate=0.5                                    |
| Conv2D          | 256     | 3           | relu       |                | kernel_regularizer=regularizers.l2(0.03)    |
| MaxPooling2D    |         | 2           |            |                | padding='same'                              |
| Dropout         |         |             |            |                | rate=0.5                                    |
| Conv2D          | 256     | 3           | relu       |                | kernel_regularizer=regularizers.l2(0.03)    |
| MaxPooling2D    |         | 2           |            |                | padding='same'                              |
| Dropout         |         |             |            |                | rate=0.5                                    |
| Flatten         |         |             |            |                |                                             |
| Dense           |         |             | sigmoid    |                | Output: 1 neuron                            |

#### CNN 5: Input Layer + Multiple Hidden Layers with Dropout and Learning Rate Adjustment

| Layer Type      | Filters | Kernel Size | Activation | Input Shape    | Additional Parameters                       |
|-----------------|---------|-------------|------------|----------------|---------------------------------------------|
| Conv2D          | 512     | 3           | relu       | (256, 256, 3)  |                                             |
| MaxPooling2D    |         | 2           |            |                | padding='same'                              |
| Dropout         |         |             |            |                | rate=0.5                                    |
| Conv2D          | 256     | 3           | relu       |                | kernel_regularizer=regularizers.l2(0.03)    |
| MaxPooling2D    |         | 2           |            |                | padding='same'                              |
| Dropout         |         |             |            |                | rate=0.5                                    |
| Conv2D          | 256     | 3           | relu       |                | kernel_regularizer=regularizers.l2(0.03)    |
| MaxPooling2D    |         | 2           |            |                | padding='same'                              |
| Dropout         |         |             |            |                | rate=0.5                                    |
| Conv2D          | 256     | 3           | relu       |                | kernel_regularizer=regularizers.l2(0.03)    |
| MaxPooling2D    |         | 2           |            |                | padding='same'                              |
| Dropout         |         |             |            |                | rate=0.5                                    |
| Flatten         |         |             |            |                |                                             |
| Dense           |         |             | sigmoid    |                | Output: 1 neuron                            |


### Performance Metrics
The performance of the models was evaluated using accuracy and loss metrics, additionally, CNN4 was evaluated using an F1 score:
| Model | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy | F1 Score | Observations                                                                                   |
|-------|---------------|-------------------|-----------------|---------------------|----------|-----------------------------------------------------------------------------------------------|
| CNN 1 | 0.4773        | 0.7526            | 0.5494          | 0.7180              | -        | Good initial performance with a simple architecture. Lacks complexity to capture intricate patterns. |
| CNN 2 | 0.3744        | 0.8161            | 0.6049          | 0.7397              | -        | Improved accuracy due to additional hidden layers, but overfitting likely due to lack of regularization. |
| CNN 3 | 1.1090        | 0.6189            | 1.0557          | 0.6176              | -        | Regularization (L2, dropout) added; performance decreased, indicating regularization may need tuning. |
| CNN 4 | 0.6682        | 0.5934            | 0.6603          | 0.6025              | 0.60     | High dropout reduced overfitting but led to underfitting, causing lower accuracy.                 |




### CNN Loss Function & Accuracy Graphs

<table style="width:100%">
  <tr>
    <td><img src="https://github.com/MartaFuentesF/capstone_project/blob/main/images_and_graphs/cnn_graphs_and_diagrams/cnn1_graph.png?raw=true" alt="CNN1 Loss Function & Accuracy" width="100%"></td>
    <td><img src="https://github.com/MartaFuentesF/capstone_project/blob/main/images_and_graphs/cnn_graphs_and_diagrams/cnn2_graph.png?raw=true" alt="CNN2 Loss Function & Accuracy" width="100%"></td>
  </tr>
  <tr>
    <td><img src="https://github.com/MartaFuentesF/capstone_project/blob/main/images_and_graphs/cnn_graphs_and_diagrams/cnn3_graph.png?raw=true" alt="CNN3 Loss Function & Accuracy" width="100%"></td>
    <td><img src="https://github.com/MartaFuentesF/capstone_project/blob/main/images_and_graphs/cnn_graphs_and_diagrams/cnn4_graph.png?raw=true" alt="CNN4 Loss Function & Accuracy" width="100%"></td>
  </tr>
</table>

### CNN4 ROC Curve with AUC score
![CNN4 ROC AUC]('https://github.com/MartaFuentesF/capstone_project/blob/main/images_and_graphs/cnn_graphs_and_diagrams/cnn4_ROC.png?raw=true')

## Summary of Results

- **CNN 1**: The initial model provided a solid baseline with relatively high training and validation accuracy. Its simple architecture allowed for quick training but limited its ability to learn more complex patterns in the data. This model showed good initial performance but lacked the complexity required for higher accuracy on more varied data.

- **CNN 2**: By adding more hidden layers, CNN 2 significantly improved training accuracy. However, the model showed signs of overfitting, as indicated by a lower validation accuracy compared to training accuracy. The lack of regularization meant that while the model could learn from the training data, it struggled to generalize to new, unseen data.

- **CNN 3**: Regularization techniques such as L2 regularization and dropout were introduced in this model to combat overfitting. Unfortunately, these adjustments led to a decrease in overall performance. This suggests that the regularization parameters were too strong, resulting in underfitting where the model couldn't adequately capture the data patterns.

- **CNN 4**: This model increased the dropout rates to further mitigate overfitting. Although this approach successfully reduced overfitting, it resulted in lower overall accuracy, suggesting that the model may have been overly regularized. This excessive regularization likely hindered the model's ability to learn effectively. The ROC curve and AUC indicate that the model's True Positive Rate is suboptimal, highlighting significant limitations in its predictive performance. Additionally, the F1 score for CNN 4 is 0.6, which implies a balanced performance between precision and recall. However, this score also reflects moderate precision and recall, suggesting that the model has significant misclassifications. In a confusion matrix, this means the model has a notable number of both false positives and false negatives, further underscoring the need for better optimization.

## Key Observations

1. **Architecture Complexity**: Increasing the complexity of the model by adding more layers (as in CNN 2) improved training performance but also introduced overfitting due to the lack of regularization.
2. **Regularization**: Adding regularization (as in CNN 3 and CNN 4) helped mitigate overfitting but required careful tuning. Excessive regularization led to underfitting, where the model could not learn effectively from the data.
3. **Model Balance**: A balance between model complexity and regularization is crucial. While complex models have higher learning capacity, they also require adequate regularization to generalize well to new data.

This analysis underscores the importance of iterative tuning and evaluation to achieve the right balance between model complexity and regularization, ensuring robust performance across both training and validation datasets.

## Technical Insights

### Convolutional Layers
The number of filters and the kernel size in convolutional layers impact the model's ability to detect features. Larger filters can capture more complex patterns, while smaller kernel sizes focus on finer details.

### Pooling Layers
MaxPooling layers help reduce the spatial dimensions of the feature maps, reducing the computational load and the risk of overfitting. Padding in pooling layers ensures that the spatial dimensions are preserved, which can be crucial for deeper networks.

### Regularization
Regularization techniques such as Dropout and L2 regularization help prevent overfitting by introducing noise during training and penalizing large weights, respectively.

### Loss Function and Optimizer
Binary cross-entropy was chosen as the loss function because the problem is a binary classification. The Adam optimizer is used for its efficiency and adaptability, providing a good balance between speed and performance.

## Challenges and Learnings

### Computational Resources
The models required substantial computational power and time to fit. Each model took five to ten hours to fit, using an M3 chip with 18GB of memory. Future work could explore cloud-based solutions.

### Quality of the Data
Given the objective of this model is to differentiate between the faces of adults and youth, it would be highly beneficial to train the model using datasets that consist exclusively of facial images. However, acquiring large datasets that meet such specific requirements poses a significant challenge.

## Future Work
Visualizing the images that were misclassified to gain some understanding of how to continue to tune the model. Future improvements could include experimenting with more advanced architectures like ResNet, implementing data augmentation, and exploring other regularization techniques. Additionally, evaluating the model using other metrics such as precision, recall, and F1-score could provide more insights into its performance.

## Conclusion

This project successfully developed and evaluated four CNN models for distinguishing between the faces of adults and youth. While the models demonstrated promising initial results, further improvements and optimizations are essential for real-world applications. Understanding the reasons behind the models' poor performance provides several key benefits:

1. **Enhanced Model Development**: By analyzing why certain models underperformed, we can better understand the impact of architectural choices, regularization techniques, and data preprocessing methods. This knowledge guides the development of more effective models in the future.

2. **Targeted Optimization**: Identifying the causes of overfitting or underfitting allows for targeted optimizations. Adjustments such as fine-tuning regularization parameters, modifying the number of layers, and improving data augmentation techniques can significantly enhance model performance.

3. **Improved Generalization**: Insights gained from poor performance help in designing models that generalize better to unseen data. By addressing the factors that lead to overfitting, we can create models that perform consistently across diverse datasets.

4. **Application Versatility**: Despite initial performance challenges, the potential applications of this model span various sectors, including healthcare for age-related diagnoses, social media for age verification, and marketing for age-targeted strategies. 

By leveraging the lessons learned from this project's initial phases, we can refine our approach and develop robust CNN models that meet the specific needs of real-world applications. The versatility and impact of these models are significant, provided that the continuous process of evaluation and optimization is maintained.


## References:
* [Getting Started with Keras]('https://keras.io/getting_started/')
* [Image Data Loading]('https://keras.io/api/data_loading/image/')
* [Image Augmentation Layers]('https://keras.io/api/layers/preprocessing_layers/image_augmentation/')
* [How to split folder of images into test/train/validation sets with stratified sampling]('https://stackoverflow.com/questions/53074712/how-to-split-folder-of-images-into-test-training-validation-sets-with-stratified')
* [tf.keras.preprocessing.image_dataset_from_directory Value Error: No images found]('https://stackoverflow.com/questions/68449103/tf-keras-preprocessing-image-dataset-from-directory-value-error-no-images-found')
* [Access images after tf.keras.utils.image_dataset_from_directory]('https://stackoverflow.com/questions/73672773/access-images-after-tf-keras-utils-image-dataset-from-directory')
* [TensorFlow 2.0 Tutorial 01: Basic Image Classification]('https://lambdalabs.com/blog/tensorflow-2-0-tutorial-01-image-classification-basics')
* [`shutil` High-level file operations]('https://docs.python.org/3/library/shutil.html')
* [Python| os.listdir() method]('https://www.geeksforgeeks.org/python-os-listdir-method/')
* [Python| os.path.join() method]('https://www.geeksforgeeks.org/python-os-path-join-method/')
* [How to Calculate Precision, Recall, F1, and More for Deep Learning Models]('https://machinelearningmastery.com/how-to-calculate-precision-recall-f1-and-more-for-deep-learning-models/')
* [Python|os.listdir()method]('https://www.geeksforgeeks.org/python-os-listdir-method/')
* [BatchDataset display images and label]('https://stackoverflow.com/questions/70456447/batchdataset-display-images-and-label')
* [stackoverflow.com]('https://stackoverflow.com/questions/66715975/class-weights-in-cnn')
* [Stack Exchange- itdexer]('https://stats.stackexchange.com/questions/207049/neural-network-for-binary-classification-use-1-or-2-output-neurons').
---
**Addendum: The Unsuccessful Approach:**

Initially, a different data preparation strategy was attempted. This approach involved manually splitting the dataset into training and validation subsets using sklearn's `train_test_split()`.

This is the desired structure for this approach:
![File_Structure](https://github.com/MartaFuentesF/capstone_project/blob/main/images_and_graphs/cnn_graphs_and_diagrams/structure_unsucessful_approach.png)

To load the images for my model, I used Keras' `image_dataset_from_directory()` function. However, this approach led to an unexpected increase of around 8,500 images. I've started investigating to pinpoint the issue and will continue to work on resolving it. Fortunately, a second approach was successful.

---


