# Machine Learning Portfolio

Welcome to my Machine Learning Portfolio! This repository showcases various projects that demonstrate my skills in tackling real-world machine learning problems using a range of algorithms, methodologies, and tools.

---

## Project List

### Question Pair Classification using BERT
- **Directory**: `Question Pair Classification`
- **Description**: The main objective of this project is to build a machine learning model to classify whether two given questions are duplicates or not, based on their semantic meaning. For this task, we used a dataset that contains pairs of questions, along with a target variable `is_duplicate` indicating whether the questions are semantically equivalent. The goal is to determine if the two questions express the same intent, which is important for applications like duplicate question detection, question answering systems, and search engine optimization.
- **Methods**: We used BERT (Bidirectional Encoder Representations from Transformers) for sequence classification in this project. The text data was preprocessed by converting to lowercase, removing non-alphabetical characters, and tokenizing using the BERT tokenizer. The dataset was split into training and test sets (80%/20%). We fine-tuned the BERT-base-uncased model for binary classification to predict whether two questions are duplicates or not. Hyperparameter tuning was applied to optimize the learning rate, batch size, and number of epochs, with the final configuration set to a learning rate of 2e-5, batch size of 32, and 2 epochs. Training was done using Hugging Face’s Trainer API for efficient model fine-tuning.
- **Key Objective**: The key objective of this project is to accurately classify pairs of questions as duplicates or not, leveraging **BERT’s transformer architecture** to capture contextual semantic meaning.

### Bank deposit forecasting project

- **Directory**: `Bank deposit forecasting project`
- **Description**: The main objective of this project is to build a model to predict whether a customer will open a term deposit in a bank. For this task, we used a dataset from the UCI Machine Learning Repository, which contains data related to direct marketing campaigns (phone calls) of a Portuguese banking institution. The goal is to predict customer responses to marketing efforts and optimize bank strategies to increase the number of term deposits.
- **Methods**: We implemented SMOTE and RandomUnderSampler to address class imbalance in the dataset. The models trained include Logistic Regression, K-Nearest Neighbors (KNN), Decision Trees, and XGBoost. Hyperparameter tuning was applied to improve the performance of these models using GridSearchCV and Hyperport.
- **Key Objective**: Accurately predict whether a customer will subscribe to a term deposit, helping banks optimize their marketing campaigns and improve customer acquisition strategies.

### Forecasting the number of passengers on international airlines

- **Directory**: `Forecasting the number of passengers on international airlines`
- **Description**: The primary objective of this project is to forecast the number of international airline passengers in units of 1,000 for a given year and month. The dataset contains a single feature: Number of passengers (labelled as Passengers), which records the number of passengers flying on international airlines each month.
The goal is to use Long Short-Term Memory (LSTM) networks, a type of recurrent neural network (RNN), to predict future passenger numbers based on historical data. LSTM is specifically chosen for its ability to capture temporal dependencies in time-series data.
- **Methods**: The model is built using LSTM, and the Adam optimizer is used for training. The model evaluation involves checking the performance based on loss and prediction accuracy over the test data.
- **Key Objective**: The main goal is to accurately forecast future passenger numbers, which could assist airlines in better understanding travel demand, optimizing flight schedules, and improving resource allocation strategies.  

### Store Demand Time Series Modeling

- **Directory**: `store-demand-time-series-modeling`
- **Description**: In this project, we develop a time series forecasting model using data from the *Store Item Demand Forecasting Challenge* dataset.
- **Methods**: Our approach includes exploring seasonality, trends, and autocorrelation in the data to predict daily sales for each item-store pair. We use five years of historical sales data to generate demand forecasts.
- **Key Objective**: Provide accurate daily demand forecasts at the item-store level to support inventory management and reduce stockouts or overstocking.

### Bank Customer Churn Prediction

- **Directory**: `bank-customer-churn-prediction`
- **Description**: This project focuses on predicting customer churn using data from the *Bank Customer Churn Prediction (DLU Course)* dataset. Our goal is to create a classification model to determine whether a customer will continue using their bank account or close it.
- **Methods**: We implemented gradient-boosting algorithms, specifically **XGBoost** and **LightGBM**, to optimize model accuracy. Additionally, the **HyperOpt** library was used for hyperparameter tuning.
- **Key Objective**: Accurately predict customer churn to help banks enhance customer retention strategies.
  
---

Each project directory contains:

- **Detailed Notebooks and Code**: Organized and well-documented code to guide through the solution process.
- **Model Insights and Results**: Key findings and performance metrics to highlight each model's effectiveness.

Feel free to explore each folder to dive into the specifics of the models, the code, and the results of each project.

---
