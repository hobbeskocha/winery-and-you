# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: winery-project
#     language: python
#     name: python3
# ---

# ## Import Libraries and Config pandas display

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.options.display.max_columns = 25
pd.options.display.max_rows = 100

# ## Import datasets from data cleaning process

winery = pd.read_csv("../data/Winery_Data_Clean.csv")
winery.dtypes

# +
winery["Date"] = pd.to_datetime(winery["Date"], format="%Y-%m-%d")
winery = winery.astype({"CustomerID": object,
                        "OrderID": object})
categorical_columns = ["CustomerSegment", "State", "ZipCode", "Division", "Region", "Channel"]
winery[categorical_columns] = winery[categorical_columns].astype("category")

winery.dtypes
# -

customer = pd.read_csv("../data/Winery_Customer.csv")
customer.dtypes

# +
customer = customer.astype({"CustomerID": object})
categorical_columns = ["CustomerSegment", "State", "ZipCode", "Division", "Region"]
customer[categorical_columns] = customer[categorical_columns].astype("category")

customer.dtypes
# -

sns.set_palette("colorblind")

# ## Descriptive Analysis

# Preliminary analysis was performed using seaborn. More detailed analysis was performed using Tableau

# ### Orders

winery.columns

# #### Baseline

plt.figure(figsize=(12, 8))
sns.boxplot(data = winery, y = "SaleAmount", x = "Channel")
plt.title('Boxplot of Sale Amount by Channel')
plt.show()

plt.figure(figsize=(12, 8))
sns.boxplot(data = winery, y = "SaleAmount", x = "CustomerSegment")
plt.title('Boxplot of Sale Amount by Customer Segment')
plt.show()

# #### Log Transform Sale Amount

winery_copy = winery.copy()
winery_copy["SaleAmount"] = winery_copy["SaleAmount"].replace(0, 0.0001)
winery_copy["SaleAmount_log"] = np.log(winery_copy["SaleAmount"])

plt.figure(figsize=(12, 8))
sns.boxplot(data = winery_copy, y = "SaleAmount_log", x = "Channel")
plt.title('Boxplot of Logged Sale Amount by Channel')
plt.show()

plt.figure(figsize=(12, 8))
sns.boxplot(data = winery_copy, y = "SaleAmount_log", x = "CustomerSegment")
plt.title('Boxplot of Logged Sale Amount by Customer Segment')
plt.show()

# ### Customers

customer.columns

customer_copy = customer.copy()
# customer_copy["SaleAmount"] = customer_copy["SaleAmount"].replace(0, 0.0001)
customer_copy["SaleAmount_log"] = np.log(customer_copy["SaleAmount"])

# #### Baseline

plt.figure(figsize=(12, 8))
sns.boxplot(data = customer_copy, y = "SaleAmount", x = "CustomerSegment")
plt.title("Boxplot of Sale Amount by Customer Segment")
plt.show()

plt.figure(figsize=(12, 8))
sns.boxplot(data = customer, y = "OrderVolume", x = "CustomerSegment")
plt.title("Boxplot of Order Volume")
plt.show()

# #### Log Tranformed

plt.figure(figsize=(12, 8))
sns.boxplot(data = customer_copy, y = "SaleAmount_log", x = "CustomerSegment")
plt.title("Boxplot of Sale Amount by Customer Segment")
plt.show()

# ### Analyzing the distribution of customer segment counts across states, regions, and divisions. Is there a considerable difference in the distribution of customer segments across the USA?

segment_by_states = winery.groupby("State", observed=False)["CustomerSegment"].value_counts()
segment_by_region = winery.groupby("Region", observed=False)["CustomerSegment"].value_counts()
segment_by_division = winery.groupby("Division", observed=False)["CustomerSegment"].value_counts()

# #### Distribution by State

# +
segment_by_states_df = segment_by_states.reset_index(name='Count')

plt.figure(figsize=(12, 8))
sns.barplot(x='State', y='Count', hue='CustomerSegment', data=segment_by_states_df)
plt.title('Customer Segments Count by State')
plt.xticks(rotation=45)
plt.show()
# -

# #### Distribution by Region

# +
segment_by_region_df = segment_by_region.reset_index(name='Count')

sns.barplot(x='Region', y='Count', hue='CustomerSegment', data=segment_by_region_df)
plt.title('Customer Segments Count by Region')
plt.show()
# -

# #### Distribution by Division

# +
segment_by_division_df = segment_by_division.reset_index(name='Count')

# plt.figure(figsize=(12, 8))
sns.barplot(x='Division', y='Count', hue='CustomerSegment', data=segment_by_division_df)
plt.title('Customer Segments Count by Division')
plt.xticks(rotation=45)
plt.show()
# -

# ### Channel sales amount by customer segments, geographic regions (states, regions, and divisions) 

# #### Total channel sales amount by customer segment

# +
channel_sales_by_segment = winery.groupby(["CustomerSegment", "Channel"], observed=False)["SaleAmount"].sum().reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(x='CustomerSegment', y='SaleAmount', hue='Channel', data=channel_sales_by_segment)
plt.title('Sum of SaleAmount by CustomerSegment and Channel')
plt.xlabel('CustomerSegment')
plt.ylabel('Sum of SaleAmount')
plt.show()
# -

# #### Total channel sales amount by Region

# +
channel_sales_by_segment = winery.groupby(["Region", "Channel"], observed=False)["SaleAmount"].sum().reset_index()

plt.figure(figsize=(12, 8))
sns.barplot(x='Region', y='SaleAmount', hue='Channel', data=channel_sales_by_segment)
plt.title('Sum of SaleAmount by Region and Channel')
plt.xlabel('Region')
plt.ylabel('Sum of SaleAmount')
plt.show()
# -

# #### Total channel sales amount by State

# +
channel_sales_by_segment = winery.groupby(["State", "Channel"], observed=False)["SaleAmount"].sum().reset_index()

plt.figure(figsize=(12, 8))
sns.barplot(x='State', y='SaleAmount', hue='Channel', data=channel_sales_by_segment)
plt.title('Sum of SaleAmount by State and Channel')
plt.xlabel('State')
plt.ylabel('Sum of SaleAmount')
plt.xticks(rotation=45)

plt.show()
# -

# #### Total channel sales amount by Division

# +
channel_sales_by_segment = winery.groupby(["Division", "Channel"], observed=False)["SaleAmount"].sum().reset_index()

plt.figure(figsize=(12, 8))
sns.barplot(x='Division', y='SaleAmount', hue='Channel', data=channel_sales_by_segment)
plt.title('Sum of SaleAmount by State and Channel')
plt.xlabel('Division')
plt.ylabel('Sum of SaleAmount')
plt.xticks(rotation=45)

plt.show()
# -

# ### Volume of orders by geographic regions, channel including segments

vol_by_region_seg = winery.groupby(["Region", "CustomerSegment"], observed=False)["OrderID"].count().reset_index(name='OrderVolume')
vol_by_division_seg = winery.groupby(["Division", "CustomerSegment"], observed=False)["OrderID"].count().reset_index(name='OrderVolume')
vol_by_channel_seg = winery.groupby(["Channel", "CustomerSegment"], observed=False)["OrderID"].count().reset_index(name='OrderVolume')

# #### Customer Segment Order volume by region 

# +
plt.figure(figsize=(12, 8))
sns.barplot(x='Region', y='OrderVolume', hue='CustomerSegment', data=vol_by_region_seg)
plt.title('Order Volume by Region and Segment')
plt.xlabel('Region')
plt.ylabel('Order Volume')
plt.xticks(rotation=45)

plt.show()
# -

# #### Customer Segment Order volume by division

# +
plt.figure(figsize=(12, 8))
sns.barplot(x='Division', y='OrderVolume', hue='CustomerSegment', data=vol_by_division_seg)
plt.title('Order Volume by Division and Segment')
plt.xlabel('Division')
plt.ylabel('Order Volume')
plt.xticks(rotation=45)

plt.show()
# -

# #### Customer Segment Order volume by channel

# +
plt.figure(figsize=(12, 8))
sns.barplot(x='Channel', y='OrderVolume', hue='CustomerSegment', data=vol_by_channel_seg)
plt.title('Order Volume by Channel and Segment')
plt.xlabel('Channel')
plt.ylabel('Order Volume')
plt.xticks(rotation=45)

plt.show()
# -

# ### Subscription preferences by customer segments, geographic regions (states, regions, and divisions) 

# #### Groupings by Segment, Region, and Division for subscription preferences

customer_seg_email = customer.groupby("CustomerSegment", observed=False)["EmailSubscr"].value_counts().reset_index(name = "SubscrCount")
customer_seg_newsletter = customer.groupby("CustomerSegment", observed=False)["NewsletterSubscr"].value_counts().reset_index(name = "SubscrCount")
customer_seg_winemakercalls = customer.groupby("CustomerSegment", observed=False)["WinemakerCallSubscr"].value_counts().reset_index(name = "SubscrCount")

customer_reg_email = customer.groupby("Region", observed=False)["EmailSubscr"].value_counts().reset_index(name = "SubscrCount")
customer_reg_newsletter = customer.groupby("Region", observed=False)["NewsletterSubscr"].value_counts().reset_index(name = "SubscrCount")
customer_reg_winemakercalls = customer.groupby("Region", observed=False)["WinemakerCallSubscr"].value_counts().reset_index(name = "SubscrCount")

customer_div_email = customer.groupby("Division", observed=False)["EmailSubscr"].value_counts().reset_index(name = "SubscrCount")
customer_div_newsletter = customer.groupby("Division", observed=False)["NewsletterSubscr"].value_counts().reset_index(name = "SubscrCount")
customer_div_winemakercalls = customer.groupby("Division", observed=False)["WinemakerCallSubscr"].value_counts().reset_index(name = "SubscrCount")

# #### Subscription preferences by Segment

plt.figure(figsize=(10, 6))
sns.barplot(x='CustomerSegment', y='SubscrCount', hue='EmailSubscr', data=customer_seg_email)
plt.title('Email Subscriptions Count by Customer Segment')
plt.xlabel('Customer Segment')
plt.ylabel('Subscriptions Count')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x='CustomerSegment', y='SubscrCount', hue='NewsletterSubscr', data=customer_seg_newsletter)
plt.title('Newsletter Subscriptions Count by Customer Segment')
plt.xlabel('Customer Segment')
plt.ylabel('Subscriptions Count')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x='CustomerSegment', y='SubscrCount', hue='WinemakerCallSubscr', data=customer_seg_winemakercalls)
plt.title('Winemaker Calls Subscriptions Count by Customer Segment')
plt.xlabel('Customer Segment')
plt.ylabel('Subscriptions Count')
plt.xticks(rotation=45)
plt.show()

# #### Subscription preferences by Region

plt.figure(figsize=(10, 6))
sns.barplot(x='Region', y='SubscrCount', hue='EmailSubscr', data=customer_reg_email)
plt.title('Email Subscriptions Count by Region')
plt.xlabel('Region')
plt.ylabel('Subscriptions Count')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x='Region', y='SubscrCount', hue='NewsletterSubscr', data=customer_reg_newsletter)
plt.title('Newsletter Subscriptions Count by Region')
plt.xlabel('Region')
plt.ylabel('Subscriptions Count')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x='Region', y='SubscrCount', hue='WinemakerCallSubscr', data=customer_reg_winemakercalls)
plt.title('WinemakerCall Subscriptions Count by Region')
plt.xlabel('Region')
plt.ylabel('Subscriptions Count')
plt.xticks(rotation=45)
plt.show()

# #### Subscription prefences by Division

plt.figure(figsize=(10, 6))
sns.barplot(x='Division', y='SubscrCount', hue='EmailSubscr', data=customer_div_email)
plt.title('Email Subscriptions Count by Division')
plt.xlabel('Division')
plt.ylabel('Subscriptions Count')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x='Division', y='SubscrCount', hue='NewsletterSubscr', data=customer_div_newsletter)
plt.title('Newsletter Subscriptions Count by Division')
plt.xlabel('Division')
plt.ylabel('Subscriptions Count')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x='Division', y='SubscrCount', hue='WinemakerCallSubscr', data=customer_div_winemakercalls)
plt.title('WinemakerCall Subscriptions Count by Division')
plt.xlabel('Division')
plt.ylabel('Subscriptions Count')
plt.xticks(rotation=45)
plt.show()
