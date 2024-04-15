# immo-eliza-ml

## Model Card: Linear Regression Model

### Overview

This model card provides details about a linear regression model trained for predicting house prices in the Belgium real estate market.

### Model Details

- **Model Name:** Linear Regression, modified-Z-score outliers
- **Model Version:** v0.1
- **Model Type:** Linear Regression
- **Date:** 23-02-2024
- **Author:** Brian Daza

### Intended Use

This model is intended to predict house prices based on a selection of features from a dataset 

### Dataset

- **Dataset Name:** properties.csv
- **Dataset Description:** Dataset was already curated and basic cleaning was performed on it

### Training Data Preprocessing

- **Data Preprocessing Techniques:** Preprocessing steps for this model included removing some numerical variables that where not present in the mayority of the data. This includes: cadastral income, terrace_sqm, garden_sqm, primary energy consumption sqm

Also zipcodes where removed for now. Can be implemented in a later version of the model as categorize. Same goes for construction year


### Model Training

- **Training Algorithm:** Training algorithm used for this model was a simple linear regression
- **Training Data:** Using train-test-split training data was used from curated dataset of an Immoweb scraoping project

### Evaluation

- **Evaluation Metrics:** R**2 scores in both training and test data where used to evaluate model performance
- **Evaluation Results:** 
R**2 for training data = 0.33
R**2 for training data = 0.39

### Limitations

- **Data Limitations:** To optimize model data still requires feature engineering, for example the postal codes are not considered, construction year and EPC needs to be adjusted per region
- **Model Limitations:** The model is based on linear regression, but the data tends to plateau at higher values, another model might be better suited for this such as logaritmic or random forest

### Deployment

Upcoming...


