# astrd-spctra

# Asteroid Taxonomy Spectra Classification

## Overview
Spectra, a fundamental tool in astrophysics, provide insights into the composition and characteristics of celestial objects. Take our sun, for instance, whose spectrum reveals the diverse energies of sunlight across different wavelengths. Yet, asteroids, those silent wanderers of the cosmos, also possess spectra, offering a window into their secrets. In this project, we delve into the intriguing world of asteroid spectra taxonomy to classify and understand these enigmatic objects.
This  machine learning project focuses on classifying asteroid taxonomy spectra using various models and techniques. The dataset comprises over 1,000 spectra from.
The primary objectives of this project are:

1. Distinguish between the X class and "non X class" asteroids.
2. Perform multi-label classification to categorize asteroids into multiple taxonomic classes.
3. Utilize unsupervised clustering techniques, specifically autoencoders, to discover hidden patterns and structures in the data.

   Classification: We utilize asteroid spectra to distinguish between various taxonomic classes, such as C, S, X, and Others. These classifications help us understand the composition and origins of asteroids.

Machine Learning: Spectral data serves as the primary input for our machine learning models. By training these models on the spectra, we can make accurate predictions and categorize asteroids into their respective taxonomic groups.

Pattern Discovery: Unsupervised clustering techniques, including autoencoders, rely on spectral data to uncover hidden patterns and structures within the asteroid dataset. This aids in identifying similarities and differences among asteroids.

In essence, asteroid spectra serve as the foundation for our data-driven approach, enabling us to classify, predict, and gain deeper insights into these celestial objects.

## Table of Contents

1. [Requirements](#requirement)
2. [Installation](#installation)
3. [Data](#data)
4. [Approach](#approach)
5. [Results](#results)


## Requirements

To run this project, you need the following dependencies:

- Python 3.x
- NumPy
- Pandas
- TensorFlow (or other deep learning framework of your choice)
- Scikit-learn (for machine learning models)
- Matplotlib (for visualization)
- Jupyter Notebook (for running provided notebooks)

## Data
The dataset used for this project can be obtained from the following sources:

Asteroid Taxonomy Spectra Data:

Data Source: Bus Taxonomy Spectra
SHA-256 Checksum: 0ce970a6972dd7c49d512848b9736d00b621c9d6395a035bd1b4f3780d4b56c6
Additional Asteroid Spectra Data:

Data Source: SMass2 Data
SHA-256 Checksum: dacf575eb1403c08bdfbffcd5dbfe12503a588e09b04ed19cc4572584a57fa97
Please download the necessary data from the provided sources and place them in the data directory of this project before running the notebooks.


## Approach
A summary of asteroid classification schemas, the science behind it and some historical context can be found here. One flow chart shows the link between miscellaneous classification schemas. On the right side the flow chart merges into a general "main group". These groups are:

C: Carbonaceous asteroids
S: Silicaceous (stony) asteroids
X: Metallic asteroids
Other: Miscellaneous types of rare origin / composition; or even unknown composition like T-Asteroids

![image](https://github.com/akvk1712/astrd-spctra/assets/127722008/daabb5da-6e38-4ba8-9fce-3b86fec45e49)

We are now set to perform some Machine Learning on the Asteroid Spectra Data! We keep it simple though: The multiclass clssificaiton problem of the Main Group classes:
C
S
X
Other
Support Vector Machine (SVM) algorithm to perform some classification tasks.
Define a neural network model using TensorFlow and Keras. The model consists of the following layers:
Normalization layer: Similar to scikit-learn's StandardScaler, it normalizes the input data.
Dense layers: These are fully connected layers responsible for learning complex patterns in the data.
ReLU (Rectified Linear Unit) activation functions are used in-between layers.
The output layer has 4 neurons with a softmax activation function for multi-class classification.

Model Architecture
The Convolutional Neural Network utilizes convolutional layers with max-pooling followed by dense layers.
The architecture includes normalization layers, Conv1D layers with ReLU activation functions, max-pooling layers, and dense layers.
The output layer uses softmax activation for multi-class classification.

 Keras Tuner was employed to optimize a Convolutional Neural Network (CNN) for asteroid taxonomy classification. Hyperparameter tuning improved the model's performance, resulting in an F1 score of approximately 0.941. This demonstrated the effectiveness of hyperparameter optimization in enhancing the CNN's classification accuracy.

 ## Results
Throughout this Project, I embarked on a comprehensive analysis of asteroid taxonomy spectra, driven by the primary objectives of distinguishing between the X class and "non-X class" asteroids, performing multi-label classification into multiple taxonomic classes, and harnessing the power of unsupervised clustering techniques using autoencoders to unveil latent patterns in the data.

## Distinguishing X Class from "Non-X Class" Asteroids

I successfully developed and trained classification models, including Support Vector Machine (SVM) algorithms and neural network models, to excel at distinguishing X class asteroids from their "non-X class" counterparts. My classifiers exhibited remarkable accuracy in identifying X class asteroids, marking a significant achievement in the realm of asteroid taxonomy studies.

## Multi-Label Classification

In the multifaceted domain of multi-label classification, I tackled the intricate task of categorizing asteroids into multiple taxonomic classes, encompassing C (Carbonaceous), S (Silicaceous), X (Metallic), and Other (Miscellaneous types or unknown compositions). Employing neural network models, including Convolutional Neural Networks (CNNs), and tapping into Keras Tuner for hyperparameter optimization, I showcased my prowess.

The outcomes of my multi-label classification models were highly promising, with F1 scores and accuracy levels reflecting my capacity to adeptly categorize asteroids into their respective taxonomic classes. This accomplishment underscores my contribution to advancing our comprehension of the diverse composition of asteroids.

## Unsupervised Clustering with Autoencoders

To delve deeper into the dataset and unearth hidden patterns and structures, I harnessed the power of unsupervised clustering techniques, particularly autoencoders. These models allowed me to transform the labyrinthine, high-dimensional spectral data into a streamlined, lower-dimensional latent space, thereby unveiling concealed structures.

While my journey through unsupervised clustering analysis is ongoing, the preliminary results suggest that autoencoders hold the key to identifying previously unnoticed patterns and groupings within the asteroid spectra data. This newfound insight has the potential to usher in significant advancements in our understanding of asteroid taxonomy.

## Hyperparameter Optimization with Keras Tuner

My decision to employ Keras Tuner for hyperparameter optimization yielded substantial dividends. The optimized Convolutional Neural Network (CNN) that I meticulously fine-tuned achieved an impressive F1 score of approximately 0.941. This outcome serves as a testament to my adeptness at enhancing the classification accuracy of asteroid taxonomy through hyperparameter tuning.

## Future Directions

My journey in this project sets the stage for future research and exploration in the captivating realm of asteroid taxonomy and spectral classification. As I move forward, my focus will revolve around refining the unsupervised clustering analysis with autoencoders to unearth even more intricate structures within the data. Additionally, I am committed to expanding the dataset and exploring advanced machine learning models to further elevate classification accuracy.

In conclusion, this project signifies a significant stride in the classification and analysis of asteroid taxonomy spectra. It represents my enduring commitment to contributing valuable insights to the captivating fields of planetary science and astronomy.
Quite a special niche here combining the mysteries of astronomy and machine learning to fuel the drive for humanity. Thanks to Astroniz i was able to tap into this and help me take inspiration from his videos.
 

