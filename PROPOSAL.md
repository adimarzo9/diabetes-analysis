# Project Proposal - Alessandro DI MARZO

## Title: "Diabetes Analysis & Forecasting"

## Category:

- Data analysis
- Statistical analysis

## Problem statement or motivaiton:

Diabetes is a chronic disease affecting millions of people worldwide, with serious health complications if not detected early. Early prediction is essential to improve patient care. However, diabetes datasets are often imbalanced, as positive cases are significantly underrepresented, making accurate prediction challenging. Machine learning offers the potential to analyze demographic and clinical features to identify individuals at risk and support timely interventions. This project aims to evaluate multiple machine learning models to determine which algorithm best predicts if a patient has diabetes or not. That could help clinicians in decision-making and improve patients'care.

## Planned approach and technologies:

This project will use Python 3.11 and a dependencies file to ensure that all required libraries will be implemented to be able to run the code correctly. I will be using the DiaBD dataset which was collected by a team of five researchers as part of a research project entitled “Unveiling Treatment Pathways : Using Symptoms and Demographics to Predict Effective Healthcare Decisions” from the United International University, in Bangladesh. This dataset consists of 5'288 patients'records, including 15 demographic and clinical features. The data will be analyze and preprocessed to be usable for the four machine learning models that will be trained; Gaussian Naïve Bayes (as a baseline), Logistic Regression, Random Forest and XGBoost models. These models will be trained and compared with each other through multiple evaluation metrics to establish which model best perform in diabetes prediction.

## Expected challenges and how to approach them:

We can expect challenges such as overfitting, class imbalance or a lack of generalization of the code. To deal with these challenges, it is important to preprocess correctly the data to make it usable with all models which could potentially help generalize the methods and make them usable for other real-world datasets.

## Success criteria:

- Clear visualisations of models'performance
- Being able to forecast diabetes predictions with reasonable evaluation metrics.
- Clean and compact coding
- The data used is useful to reach the goal of the project.

## Stretch goals:

- We could imagine extending the projects to multiple real-world dataset and compare them to learn new patterns and enhance diabetes predictions.
- We could imagine adding more features in the analysis.