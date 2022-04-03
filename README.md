# Neural_Network_Charity_Analysis_PT
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
* Preprocessing datasets and creating a predictive binary classifier with Neural Networks and Deep Learning Models utilizing Python's TensorFlow and Pandas modules.

## Overview

The goal of this study was to investigate and develop neural networks in Python using TensorFlow. Neural networks are a type of Machine Learning that can detect patterns and features in a dataset. Neural networks are made up of layers of neurons that may execute independent calculations and are modeled after the human brain. Image recognition is a wonderful example of a Deep Learning Neural Network. The neural network will calculate, link, weigh, and deliver an encoded category result to determine whether the image represents a "dog," "cat," or George Washington, as illustrated in the image below.

We learned the following during this module:

* How to Create a Simple Neural Network
* Prepare/process the datasets
* Make a set of training and testing materials.
* Check the model's correctness.
* To improve the model, add more neurons and hidden layers.
* Choose the best model for our dataset.

A philanthropic foundation, AlphabetSoup, is looking for a quantitative, data-driven solution to help evaluate which organizations are worthy of donations and which are "high-risk." Not every donation AlphabetSoup has made has had an impact in the past, since there have been applicants who have received funds and subsequently vanished. Beks, an AlphabetSoup data scientist, is responsible for measuring the effects of each gift and vetting the recipients to ensure that the company's funds are spent wisely. To fulfill this requirement, we must assist Beks in developing a binary classifier that can predict if an organization will be successful in receiving money. To assess the incoming data and create unambiguous decision-making outputs, we use Deep Learning Neural Networks.

## Results
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
We used a CSV file that contained over 34,000 organizations that had previously received donations. This dataset contained the following information.

More than 34,000 organizations that have received support from Alphabet Soup over the years are included in this CSV. A number of columns in this dataset record metadata on each organization, including the following:


* EIN and NAME—Identification columns
* APPLICATION_TYPE—Alphabet Soup application type
* AFFILIATION—Affiliated sector of industry
* CLASSIFICATION—Government organization classification
* USE_CASE—Use case for funding
* ORGANIZATION—Organization type
* STATUS—Active status
* INCOME_AMT—Income classification
* SPECIAL_CONSIDERATIONS—Special consideration for application
* ASK_AMT—Funding amount requested
* IS_SUCCESSFUL—Was the money used effectively


## Data Preprocessing

In order to construct, train, and test the neural network model, we first had to preprocess the data. For the Data Preprocessing section, you'll need to do the following:

* During the preprocessing stage, the EIN and NAME columns were removed because they were unnecessary.
* The IS SUCCESSFUL column was the target variable, and we binned APPLICATION
* TYPE and classified all unique values with fewer than 500 records as "Other."
* STATUS, ASK AMT, APPLICATION TYPE, and so on were added as features to the remaining 43 variables.

## Compiling, Training and Evaluating the Model

We used the following parameters to compile, train, and test the model after preprocessing the data:

The initial model had a total of 8891 parameters with 2 hidden layers and 1 output layer.

![Evaluating the Model D2](https://user-images.githubusercontent.com/93852380/161408517-8e58c2c0-b128-4d4c-a3e3-ce0949daa7af.png)


The second model had a total of 25562 parameters with 4 hidden layers and 1 output layer.

![Evaluating the Model D3](https://user-images.githubusercontent.com/93852380/161408580-4f9a0e72-f20e-4827-a1f5-9f037e80d941.png)

The third model had a total of 1719041 parameters with 3 hidden layers and 1 output layer.

![Evaluating the Model D3-1](https://user-images.githubusercontent.com/93852380/161408589-456c7861-ecd6-4249-8385-cfcce25a3b97.png)


## Summary

In conclusion, our model and multiple optimizations failed to produce the intended result of greater than 80%. The modifications were small and did not improve above 19 basis points with the variations of lengthening the epochs, eliminating variables, adding a 3rd hidden layer (done offline in Optimization attempt #4), and/or increasing/decreasing the neurons. Other Machine Learning algorithms did not produce any better results when tested. Random Forest Classifier, for example, has a prediction accuracy rating of 70.80%, which is 2.11 percent lower than the Deep Learning Neural Network model's accuracy rate (72.33 percent ).

Overall, Neural Networks are extremely complex, and finding the ideal configuration to function with this dataset might necessitate trial and error or several iterations.

## Resources

* Software: Python 3.7.9, Anaconda 4.9.2 and Jupyter Notebooks 6.1.4
* Libraries: Scikit-learn, Pandas, TensorFlow, Keras

