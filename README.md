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
