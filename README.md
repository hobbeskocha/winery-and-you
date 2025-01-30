# Winery and You!

![Winery](artifacts/winery.png)

This is a collaborative personal portfolio project between [Ayush Shrestha](https://www.linkedin.com/in/ayush-yoshi-shrestha/) and [Hyewon Jeong](https://www.linkedin.com/in/jeonghyewon/). The dataset used is on sales order data for a national winery, accessible [here](data/). The dataset was provided to us by [Dr. Oliver J. Rutz](https://foster.uw.edu/faculty-research/directory/oliver-rutz/) of the University of Washington, Seattle.

## Instructions

For easy version control purposes, the source code is stored as converted Python scripts, but HTML files of the source Python notebooks can be found [here](src/html/). Alternatively, the CLI commands [here](cli-commands-reference.md) can convert the Python scripts back into notebooks.

Additionally, a live demo website has been deployed [here](https://winery-and-you.streamlit.app/), allowing you to input customer details and receive real-time predictions on customers' subscription preferences. The source code can be found [here](src/backend/) for the backend and [here](src/frontend/) for the frontend.

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

A supplemental dataframe was also created, aggregating data to capture distinct customer information. This includes details such as each customer's State, Total Order Volume, and Total Sale Amount. This additional dataframe provides a comprehensive view of customer behavior, which is critical for subsequent analysis and modeling efforts.

## Exploratory Data Analysis

Some preliminary EDA was performed in Seaborn and Matplotlib to look at the general shape and distribution of the data. This was expanded upon in a [packaged Tableau workbook](src/winery-dashboard.twbx), yielding novel understanding of customer and order sales figures across the United States as well as over the lifetime of the dataset. A screenshot is shown below:

![Winery Dashboard](artifacts/winery-dashboard.png)

The dashboard showcases:

- _Geographic Map_: An interactive map showing where customers reside in the United States, with filtering capabilities that dynamically update the other charts in the dashboard based on the selected state.
- _Line Charts_: Display trends in total sales and order volume over the life of the dataset.
- _Stacked Bar Charts_: Illustrate the distribution of customer segments within each geographic division.

The dashboard additionally allows for dynamic filtering by customer segment, region, division, and month-year.

## Modeling & Results

Since this analysis was focussed on customer preferences, the supplemental Customer dataframe was used as the primary dataset for training and validation. To ensure robust model evaluation and identify the best-performing model, an initial train-test split was configured. This allowed for effective training and subsequent assessment of the models. A Scikit-learn pipeline was then created to preprocess the dataset, including scaling numeric features and one-hot encoding categorical features.

Predictive analysis was conducted using Statsmodels and Scikit-learn to develop binary classification models aimed at predicting customers' subscription preferences for email, newsletter, and direct calls. Logistic regression and random forest were selected as the primary models for this analysis due to their ability to provide insights into the importance of each independent variable through coefficients in logistic regression and feature importances in random forest. These insights are crucial for identifying the key factors that the winery can leverage to enhance customer engagement.

### Logistic Regression (Logit)

Three logit models were trained for the three subscription channels, producing the following results:

#### Email Logit

The Email logit model generated the following metrics against the validation set:
![plot1](artifacts/email-logit-cm.png)
![plot2](artifacts/email-logit-roc.png)

- Accuracy: 93.86
- Precision: 86.94 
- Recall: 95.28
- F1-Score: 90.92
- AUC: 0.9745

Additionally, the following features (with coefficients) were statistically significant at the 5% level:

- Order Volume: -0.43
- Sale Amount: 0.11 
- High Roller: 2.49
- Wine Enthusiast: 3.03
- Middle Atlantic Division: -0.27
- West South Central Division: -0.31
- Newsletter Subscriber: 3.26
- Winemaker Call Subscriber: 5.07

#### Direct (Winemaker) Call Logit

The Direct Call logit model generated the following metrics against the validation set:
![plot1](artifacts/winemaker-logit-cm.png)
![plot2](artifacts/winemaker-logit-roc.png)

- Accuracy: 91.08
- Precision: 96.50
- Recall: 80.12
- F1-Score: 87.55
- AUC: 0.9100

Additionally, the following features (with coefficients) were statistically significant at the 5% level:

- Order Volume: 0.35
- Luxury Estate: -0.29
- West North Central Division: 0.24 
- Newsletter Subscriber: 0.36
- Email Subscriber: 5.02

#### Newsletter Logit

The Newsletter logit model generated the following metrics against the validation set:
![plot1](artifacts/newsletter-logit-cm.png)
![plot2](artifacts/newsletter-logit-roc.png)

- Accuracy: 65.26
- Precision: 73.86
- Recall: 73.02
- F1-Score: 73.44
- AUC: 0.7282

Additionally, the following features (with coefficients) were statistically significant at the 5% level:

- Sale Amount: -0.06
- High Roller: -0.32
- Winemaker Call Subscriber: 0.38
- Email Subscriber: 3.48

### Random Forest (RF)

Similarly, three RF models were trained for the three subscription channels, with the following results:

#### Email RF

The Email RF model generated the following metrics against the validation set:

- Accuracy: 95.29
- Precision: 87.87
- Recall: 99.08
- F1-Score: 93.14

Additionally, these are the feature importances of the variables:
![plot1](artifacts/email-rf-fi.png)

With the following variables having importances at or above the 75th percentile:

- Order Volume
- Wine Enthusiast
- Newsletter Subscriber
- Winemaker Call Subscriber

#### Direct (Winemaker) Call RF

The Direct Call RF model generated the following metrics against the validation set:

- Accuracy: 91.79
- Precision: 99.93
- Recall: 79.07
- F1-Score: 88.28

Additionally, these are the feature importances of the variables:
![plot1](artifacts/winemaker-rf-fi.png)

With the following variables having importances at or above the 75th percentile:

- Order Volume
- Wine Enthusiast
- Newsletter Subscriber
- Email Subscriber

#### Newsletter RF

The Newsletter RF model generated the following metrics against the validation set:

- Accuracy: 66.14
- Precision: 70.52
- Recall: 83.40
- F1-Score: 76.42

Additionally, these are the feature importances of the variables:
![plot1](artifacts/newsletter-rf-fi.png)

With the following variables having importances at or above the 75th percentile:

- Order Volume
- Sale Amount
- Winemaker Call Subscriber
- Email Subscriber

## Model Deployment and Live Demo Website

To improve the usability of the trained models, a live demo website was built using Streamlit for the frontend and FastAPI for the backend. The website can be accessed [here](https://winery-and-you.streamlit.app/).

Making use of Streamlit's intuitive API, an interactive website was created to allow users to input a customer's details and get back predictions on their subscription preferences. This frontend application was then deployed on Streamlit Community Cloud, for easy public access.

Similarly, FastAPI was used to build the backend. It includes logic to pre-process the inputted customer details before they are passed to the trained models, as well as select the most accurate model (between Logistic Regression and Random Forest) to be used for the predictions. The backend was then containerized using Docker and deployed on Google Cloud Platform (GCP), making use of the GCP Container Registry and Cloud Run services.

The trained models themselves were packaged, exported, and uploaded to buckets on GCP Cloud Storage, which the backend downloads and uses in order to make the predictions when API requests come from the frontend.

## Findings & Recommendations

Based on the metrics, the predictive models for Email and Direct Call subscriptions demonstrate strong performance, while the models for Newsletter subscriptions are solid but offer more room for improvement.

An analysis of the coefficients and feature importances from the logistic regression and random forest models, respectively, reveals a noteworthy outcome. Specifically, in a focal subscription model, whether a customer is subscribed to one of the other two channels has a significant feature importance and a large positive effect on the likelihood of subscribing to the focal channel. This indicates a cross-channel marketing effect.

With this in mind, we can offer the following recommendations:

**Cross-channel Promotion**:

Given the identified cross-channel marketing effect, the winery should leverage cross-channel promotions through targeted messaging to enhance customer engagement. This can be implemented through a tailored approach to maximize impact. Based on the model results, we recommend the following:

- _Enhancing Marketer Calls_: Encourage the sales team to promote the benefits of Email subscriptions to Direct Call subscribers during their next interaction. The high feature importance and positive effect of Direct Call subscriptions on Email subscriptions suggest this strategy could be highly effective.
- _Email Promotions_: Develop promotional email campaigns highlighting the benefits of Direct Call and Newsletter subscriptions to current Email subscribers. Since Email subscriptions have a positive effect and high feature importance for both Direct Call and Newsletter subscriptions, this approach can drive cross-channel engagement.

By implementing these tailored cross-channel promotions, the winery can improve customer engagement across multiple marketing channels and enhance its marketing ROI.

**Emphasis on High-importance Features**:

Given that Direct Call subscription, Email subscription, and Sale Amount are consistently high-importance features across multiple models, the winery should prioritize these factors in its marketing efforts.

This could include:

- _Enhanced Call Engagement_: Invest in improving the training of the sales team responsible for marketer calls, ensuring a pleasant and seamless customer experience. Additionally, consider offering personalized and time-limited promotions to those subscribed to Direct Calls.
- _Email Campaign Improvements_: Strengthen email campaigns by incorporating personalized offers and value-added content. Examples of personalized offers could be welcome gifts, birthday discounts, and re-engagement strategies. Meanwhile, value-added content might include winery history, invitations for on-site visits, and wine pairing tips, all designed to create a more personal connection with customers.
- _Upselling Opportunities_: Recognizing that customers who spend more are more engaged, the winery should explore upselling tactics such as bundling products, offering volume discounts, or promoting relevant accessories and merchandise.

By integrating these recommendations into its business strategy, the winery will be better positioned to boost customer engagement across its marketing channels, leading to increased customer satisfaction, loyalty, and overall revenue.
