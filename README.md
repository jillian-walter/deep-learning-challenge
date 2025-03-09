# deep-learning-challenge
Challenge for Module 21 - Neural Networks/Deep Learning

This challenge utilizes data from Alphabet Soup, a non-profit that helps organizations receive funding. The data contains historical information on 34,000 organizations that have received funding from Alphabet Soup, with information on Applicant Type, Industry Sector, Govt Org Classification, Use case for funding, Organization Type, Active Status, Income Classification, Special Considerations for applications, Funding amount requested **(numerical)**, and Whether or not the money was used effectively **(binary)**. A deep learning model will be used to predict the binary classifier (was/was not successful) for future funding projects based on historical data. 

# Data Upload and Preparation: 

As with other projects, we start by importing our dependencies, the necessary libraries to create our deep learning model and perform our analysis. These libraries are listed below:

- from sklearn.model_selection import test_train_split
- from sklearn.preprocessing import StandardScaler
- import pandas as pd
- import tensorflow as tf

Our data is imported as a CSV from a static edx.com site ("https://static.bc-edx.com/data/dl-1-2/m21/lms/starter/charity_data.csv") and transformed into a dataframe using Pandas.

We start by dropping non-essential columns such as EIN (ID # per organization) and NAME (name of each organization) as this differs by organization and will not tell us anything useful in terms of success. We then preview the number of unique values for each of the other columns and determine that there are 17 types of applications, 71 different Income Classifications, 5 different use cases and other determining variables (see list below). 

**Column Types & Unique values:**
APPLICATION_TYPE            17
AFFILIATION                  6
CLASSIFICATION              71
USE_CASE                     5
ORGANIZATION                 4
STATUS                       2
INCOME_AMT                   9
SPECIAL_CONSIDERATIONS       2
ASK_AMT                   8747
IS_SUCCESSFUL                2


For columns like Application Type and Income Classification where there are a significant amount of unique variables, it is useful to reduce noise by setting a cutoff value for uniques with low counts, bucketing them into "Other". Within Application type, T3 is the most common at 27k applications, followed by T4 (1.5k), while there are other Application types with only 1-3 unique values. Within Classification, C1000 is the most common at 17,326 organizations followed by C2000 at 6,074, while there are other Income Classification types that only have 1 unique value. Because of this, we condense the columns "Application Type" and "Classification Type" to have anything with less than 500 values bucketed into "Other". 

As a final preparation step, we use the function **get_dummies** to convert all categorical data to numeric. Once the data is fully prepped, we use the **train_test_split** function from the sklearn.linear_model library to split the X and Y variables into Training and Testing groups. This is important as the model is built and trained using the Training dataset, but tested for accuracy using data it has never seen before in the Testing dataset.

# Creating and Evaluating the Model:


Overview of the analysis: Explain the purpose of this analysis.

Results: Using bulleted lists and images to support your answers, address the following questions:

Data Preprocessing

What variable(s) are the target(s) for your model?
What variable(s) are the features for your model?
What variable(s) should be removed from the input data because they are neither targets nor features?
Compiling, Training, and Evaluating the Model

How many neurons, layers, and activation functions did you select for your neural network model, and why?
Were you able to achieve the target model performance?
What steps did you take in your attempts to increase model performance?
Summary: Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.


