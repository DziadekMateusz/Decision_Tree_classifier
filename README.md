## Project Overview

This project implements a **Decision Tree classifier** to predict customer churn using a banking customer dataset. The poject covers the complete machine learning implementation, including data preprocessing, model evaluation, hyperparameter selection, training, testing and result visualization.

### Features

* Loads customer data from an Excel dataset.
* Preprocesses the data by:

  * removing unnecessary identifier columns,
  * applying one-hot encoding to categorical features.
* Splits the dataset into training and testing sets using stratified sampling.
* Evaluates Decision Tree models with different `max_depth` values using **Repeated Stratified K-Fold Cross-Validation**.
* Selects the optimal tree depth based on the highest average cross-validation accuracy.
* Trains the final Decision Tree classifier using the selected hyperparameter.
* Evaluates model performance on an independent test set.
* Generates several visualizations, including:

  * the complete Decision Tree structure,
  * decision boundaries for two selected features,
  * histogram of cross-validation accuracy scores,
  * confusion matrix for the final model.

The project demonstrates both the practical implementation of a Decision Tree classifier in Python and the use of standard machine learning evaluation techniques provided by scikit-learn.
