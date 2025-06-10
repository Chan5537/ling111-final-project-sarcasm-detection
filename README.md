# Sarcasm Detection in Reddit Comments

This repository showcases our LING 111 Final Project. <br>
Problem description: Taking a sarcasm detection model trained on Reddit data, we examine whether or not the model predicts as well on sarcasm in news headlines. <br>
Dataset source: https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection <br>
Model forked from Sarcasm Detection by NamanJain

## Description of each notebook
1. prepare-data-csv.ipynb: Using raw data source to create usable data CSVs
2. Data cleaning and EDA.ipyb: Cleaning text and some Exploratory data analysis on our data
3. Modelling.ipynb: Original CNN model for text classification task, trained on Reddit comments
4. Misclassified Comparison: Creates a spreadsheet which shows counts of how many times the model misclassified each headline

## Dataset details
FILL OUT DETAILS LATER

<table>
  <tr>
    <th></th>
    <th>Sarcastic (1)</th>
    <th>Not sarcastic (0)</th>
  </tr>
  <tr>
    <td>Train</td>
    <td>400000</td>
    <td>400000</td>	
  </tr>
  <tr>
    <td>CV</td>
    <td>50000</td>
    <td>50000</td>
  </tr>
  <tr>
    <td>Test</td>
    <td>50000</td>
    <td>50000</td>
  </tr>
  <tr>
    <td>Total</td>
    <td>500000</td>
    <td>500000</td>		
  </tr>
</table> 

## Modelling
We've used 1D CNN model to extract features from raw texts and make classifications.

### Model 1: Using only content features
Predictions made using only content features extracted from 1D CNN. Model architecture is as follows:
<p align="center">
  <img src="https://github.com/NamanJain2050/sarcasm-detection/blob/master/images/model_01.png" alt="model_01"/>
</p>
<b> Results of this model are as follows: </b>
<p align="center">
  <img src="https://github.com/NamanJain2050/sarcasm-detection/blob/master/images/model_1_cnf.png" alt="model_01"/>
</p>
The Reddit model achieved an F1-score of ---- and we were able to classify ---- of sarcastic news headlines correcly.

## Summary of results
<p align="center">
  <img src="https://github.com/NamanJain2050/sarcasm-detection/blob/master/images/summary.png" alt="summary"/>
</p>

## Conclusions
We've seen that adding emotion and sentiment features from pretrained models have <b> degraded </b> our results. <br>
Possible reason(s):
1. Models were trained on much smaller datasets as compared to our SARC dataset
