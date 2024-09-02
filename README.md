# Winery and You!

This is a collaborative personal portfolio project between [Hyewon Jeong](https://www.linkedin.com/in/jeonghyewon/) and [Ayush Shrestha](https://www.linkedin.com/in/ayush-yoshi-shrestha/). The dataset used is on sales order data for a national winery, accessible [here](data/). The dataset was provided to us by [Dr. Oliver J. Rutz](https://foster.uw.edu/faculty-research/directory/oliver-rutz/) of the University of Washington, Seattle.

## Instructions
For easy version control purposes, the source code is stored as converted Python scripts, but HTML files of the source Python notebooks can be found [here](src/html/).

## Business Problem
In the face of a highly competitive, niche industry, wineries must leverage data-driven strategies to maintain their market position. Failure to do so can lead to reduced customer satisfaction and increased churn as competing wineries gain an edge. Therefore, it is essential for wineries to thoroughly analyze their existing data to gain deeper insights into customer behavior and optimize engagement and loyalty.

This project is designed to support such a winery by analyzing sales data from customers across the United States. Through predictive modeling, we aim to provide the winery with actionable insights into the factors that drive high engagement with their three subscription-based marketing channels: email, newsletter, and direct calls. Additionally, we deliver the predictive models themselves that enable the winery to swiftly adapt to customer preferences, thereby enhancing customer loyalty and driving sustained business growth.

## Dataset Overview
The original dataset contains over 65,000 records detailing individual orders by the winery's customers. Key attributes include:
- _Customer and Order IDs_: Unique identifiers for customers and their respective orders.
- _Order Date_: The date when each order was placed.
- _Regional Attributes_: Geographic information including Zip Code and State.
- _Current Subscription Status_: Subscription status to the winery's three marketing channels - Email, Newsletter, and Direct Calls.
- _Sale Amount per Order_: The monetary value of each order, along with the marketing channel through which the sale originated.
- _Pre-defined Customer Segment_: A classification of customers into one of four segments — Casual Visitor, Wine Enthusiast, High Roller, and Luxury Estate.

## Data Cleaning & Transformation
The initial data cleaning of the dataset was carried out using Pandas, focusing on improving data quality and preparing the dataset for analysis. Key steps included:
- _Handling Missing Values_:
    - We investigated the possibility of imputing missing values, such as those in the State column, from other records associated with the same customer.
    - Since imputation was not feasible, missing values were dropped, resulting in a reduction of approximately 1% of the overall dataset.
- _Removing Erroneous Data_:
    - Data anomalies, such as negative Sale Amounts and invalid US State codes, were identified and removed. This process affected about 1.5% of the dataset.
    
In addition to cleaning, some preliminary feature engineering was performed, which included:
- _Data Type Customization_:
    - Adjusting data types for greater accuracy and utility, such as converting IDs from integers to objects and changing Email Subscription from float to boolean.
- _Geographic Mapping_:
    - Enhancing the granularity of geographic data by mapping US States to their respective US Regions and Divisions.

A supplemental dataframe was also created, aggregating data to capture distinct customer information. This included details such as each customer's State, Total Order Volume, and Total Sale Amount. This additional dataframe provides a comprehensive view of customer behavior, which is critical for subsequent analysis and modeling efforts.


## Exploratory Data Analysis
Some preliminary EDA was performed in Seaborn and Matplotlib, to look at the general shape and distribution of the data. This was expanded upon in a [packaged Tableau workbook](src/winery-dashboard.twbx), yielding novel understanding of customer and order sales figures across the United States as well as over the lifetime of the dataset.

The Tableau dashboard showcases:
- _Geographic Map_: An interactive map showing where customers reside in the United States, with filtering capabilities that dynamically update the other charts in the dashboard based on the selected state.
- _Line Charts_: Display trends in total sales and order volume over the life of the dataset.
- _Stacked Bar Charts_: Illustrate the distribution of customer segments within each geographic division.

The dashboard additionally allows for dynamic filtering by customer segment, region, division, and month-year.

## Modeling & Results
The predictive analysis is performed using Statsmodels and Scikit-learn to build predictive models for customers' subscription preferences of email, newsletter, and winemaker calls. The models used were Logistic Regression and Random Forest, namely becuase they offered insight into the weight/importance of each independent variable via coefficients and feature importances. This includes generating models metrics like accuracy, precision, and AUC to assess the predictive strength of the models. Confusion matrices were also generated to get a better sense of which cases that particular models were performing well on.
We then went one step further and calculated the lift of the subscriptions. This allowed generating marginal response and positive response curves to understand how customers would respond to solicitation for subscription.

### Logit

### Random Forest

## Recommendations