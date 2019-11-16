# Plagiarism Project, Machine Learning Deployment

The project is pursued as a part of Machine Learning Engineer Nanodegree by Udacity. A great deal of code (initial commit of this repository) was developed by the Udacity team and in not my original work.

This repository contains code and associated files for deploying a plagiarism detector using AWS SageMaker.

## Project Overview

In this project, I completed a plagiarism detector that examines a text file and performs binary classification; labeling that file as either *plagiarized* or *not*, depending on how similar that text file is to a provided source text. Detecting plagiarism is an active area of research; the task is non-trivial and the differences between paraphrased answers and original work are often not so obvious.

A strong simplification of the task is the requirement to detect plagiarism from a single source (a Wikipedia article for a respective topic).

This project is broken down into three main notebooks:

**Notebook 1: Data Exploration** (entirely provided by Udacity)
* Load in the corpus of plagiarism text data.
* Explore the existing data features and the data distribution.

**Notebook 2: Feature Engineering**  (partially provided by Udacity)

* Clean and pre-process the text data.
* Define features for comparing the similarity of an answer text and a source text, and extract similarity features.

Generated types of features:

1. Containment measure based on ngramms of different length. The containment measure assumes we know the source we want to find similarities to. This is true, if the question is whether the answer is plagiarized from a Wikipedia article about this topic. A broader question, e.g. "Is this answer plagiarized from *any* source?" makes calculation of containment infeasible. 

2. Longest Common Subsequence of words (implemented with dynamic programming)


* Select "good" features, by analyzing the correlations between different features.

The problem at hand does not have a very large number of features with high correlation. High correlation deteriorate performance of linear nonregularized estimators. I considered two approaches:

1. feature selection prior to modelling. Droping one of the highly correlated features is one approach. Without clear understanding which of the two to drop (or even which of the many to drop), I decided to involve target variable to select features for univariate selection. This approach kept containment features with n-grams with 2 to 6. Some of them have perfect correlation, which would not work with linear model (matrix inversion is not possible).

2. regularized learners, e.g. l(1) SVM

* Create train/test `.csv` files that hold the relevant features and class labels for train/test data points.

**Notebook 3: Train and Deploy Your Model in SageMaker** (partially provided by Udacity)

* Upload train/test feature data to S3.
* Define a binary classification model and a training script.
* Train your model and deploy it using SageMaker.
* Evaluate your deployed classifier.

Following the principle 'try the simplest thing first', I used a sagemaker linear learner in a container instead of building a custom pytorch or sklearn model. This approach achieved 100% accuracy on the test data. 


---