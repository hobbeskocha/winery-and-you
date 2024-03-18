# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ## Import libraries and config Pandas display

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.options.display.max_columns = 25
pd.options.display.max_rows = 100

# ## Import Dataset

# #### Read CSV

winery = pd.read_csv("../data/Winery_Data_csv.csv")
winery.head()

# #### Print basic attributes

print(winery.dtypes, "\n")
print("Dataframe shape:", winery.shape, "\n")
print(winery.info())

# ## Data cleaning & transformation

# ### Cleaning

# #### Check for NAs

winery.columns = winery.columns.str.replace(' ', '')
winery.rename(columns={"Winemakercall": "WinemakerCallSubscr"}, inplace = True)
print(winery.isna().sum())

# #### Investigate NA Orders

orders_na_condition = (winery["Orders2008"].isna()) | (winery["Orders2009"].isna()) | (winery["Orders2010"].isna())
winery_order_nas = winery.loc[orders_na_condition, :]
(winery_order_nas["CustomerID"].value_counts() > 1).sum()

# Check if all customers with NA order have others orders to impute from

# +
full_counts = winery["CustomerID"].value_counts() > 1
full_counts = full_counts[full_counts]
multi_order_custs = full_counts.index

set(winery_order_nas["CustomerID"].values).issubset(set(multi_order_custs))
# -

# Identify if there are the customers that we can impute from

list(set(winery_order_nas["CustomerID"].values).intersection(set(multi_order_custs)))

# #### Investigate NA States

state_na_condition = (winery["State"].isna())
winery_state_nas = winery.loc[state_na_condition, :]
winery_state_nas.sample(5)

# #### Drop NAs

winery = winery.dropna(ignore_index = True)     # ignore_index resets the row labels to 0 -> n-1 after dropping NA rows
print(winery.isna().sum(), "\n")
print("Shape", winery.shape)

# #### Configure column types

# +
winery["Date"] = pd.to_datetime(winery["Date"], format="%d-%b-%y")
winery = winery.astype({"CustomerID": object,
                        "OrderID": object,
                        "Orders2008": int,
                        "Orders2009": int,
                        "Orders2010": int, 
                        "EmailSubscr": bool,
                        "NewsletterSubscr": bool, 
                        "WinemakerCallSubscr": bool, 
                        "ZipCode": object})

categorical_columns = ["CustomerSegment", "State"]
winery[categorical_columns] = winery[categorical_columns].astype("category")

winery.dtypes
# -

# #### Filter for valid US states

us_states = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS',
                          'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY',
                          'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY']
winery = winery[winery["State"].isin(us_states)]

# #### Remove instances of negative sales

print(winery.describe()[["SaleAmount", "EmailSales", "NewsletterSales", "TastingRoomSales", "WinemakerCallSales"]], "\n")
print("Shape", winery.shape)

negative_sales_condition = (winery["SaleAmount"] < 0) | (winery["EmailSales"] < 0) | (winery["NewsletterSales"] < 0)\
      | (winery["WinemakerCallSales"] < 0) | (winery["TastingRoomSales"] < 0)
print("Number of negative sale rows:", len(winery[negative_sales_condition]))

# +
sales_condition = (winery["SaleAmount"] >= 0) & (winery["EmailSales"] >= 0) & (winery["NewsletterSales"] >= 0) & (winery["WinemakerCallSales"] >= 0) & (winery["TastingRoomSales"] >= 0)
winery = winery[sales_condition]

print("Number of rows after removing negative sales:", len(winery))
# -

# ### Transformation

# #### Map states to division

# +
state_to_division = {
    'AL': 'East South Central',
    'AK': 'Pacific',
    'AZ': 'Mountain',
    'AR': 'West South Central',
    'CA': 'Pacific',
    'CO': 'Mountain',
    'CT': 'New England',
    'DE': 'South Atlantic',
    'FL': 'South Atlantic',
    'GA': 'South Atlantic',
    'HI': 'Pacific',
    'ID': 'Mountain',
    'IL': 'East North Central',
    'IN': 'East North Central',
    'IA': 'West North Central',
    'KS': 'West North Central',
    'KY': 'East South Central',
    'LA': 'West South Central',
    'ME': 'New England',
    'MD': 'South Atlantic',
    'MA': 'New England',
    'MI': 'East North Central',
    'MN': 'West North Central',
    'MS': 'East South Central',
    'MO': 'West North Central',
    'MT': 'Mountain',
    'NE': 'West North Central',
    'NV': 'Mountain',
    'NH': 'New England',
    'NJ': 'Middle Atlantic',
    'NM': 'Mountain',
    'NY': 'Middle Atlantic',
    'NC': 'South Atlantic',
    'ND': 'West North Central',
    'OH': 'East North Central',
    'OK': 'West South Central',
    'OR': 'Pacific',
    'PA': 'Middle Atlantic',
    'RI': 'New England',
    'SC': 'South Atlantic',
    'SD': 'West North Central',
    'TN': 'East South Central',
    'TX': 'West South Central',
    'UT': 'Mountain',
    'VT': 'New England',
    'VA': 'South Atlantic',
    'WA': 'Pacific',
    'WV': 'South Atlantic',
    'WI': 'East North Central',
    'WY': 'Mountain',
}

winery["Division"] = winery["State"].map(state_to_division)
winery.sample(5)[["CustomerID", "OrderID", "State", "Division"]]
# -

# #### Map states to region

# +
state_to_region = {
    'AL': 'South',
    'AK': 'West',
    'AZ': 'West',
    'AR': 'South',
    'CA': 'West',
    'CO': 'West',
    'CT': 'Northeast',
    'DE': 'South',
    'FL': 'South',
    'GA': 'South',
    'HI': 'West',
    'ID': 'West',
    'IL': 'Midwest',
    'IN': 'Midwest',
    'IA': 'Midwest',
    'KS': 'Midwest',
    'KY': 'South',
    'LA': 'South',
    'ME': 'Northeast',
    'MD': 'South',
    'MA': 'Northeast',
    'MI': 'Midwest',
    'MN': 'Midwest',
    'MS': 'South',
    'MO': 'Midwest',
    'MT': 'West',
    'NE': 'Midwest',
    'NV': 'West',
    'NH': 'Northeast',
    'NJ': 'Northeast',
    'NM': 'West',
    'NY': 'Northeast',
    'NC': 'South',
    'ND': 'Midwest',
    'OH': 'Midwest',
    'OK': 'South',
    'OR': 'West',
    'PA': 'Northeast',
    'RI': 'Northeast',
    'SC': 'South',
    'SD': 'Midwest',
    'TN': 'South',
    'TX': 'South',
    'UT': 'West',
    'VT': 'Northeast',
    'VA': 'South',
    'WA': 'West',
    'WV': 'South',
    'WI': 'Midwest',
    'WY': 'West',
}

winery["Region"] = winery["State"].map(state_to_region)
winery.sample(5)[["State", "Region", "Division"]]

# -

# #### Add new "Channel" column to map orders to their respective channel

# +
def determine_channel(row):
    if row['EmailSales'] > 0:
        return "Email"
    elif row['NewsletterSales'] > 0:
        return "Newsletter"
    elif row['TastingRoomSales'] > 0:
        return "TastingRoom"
    else:
        return "WinemakerCall"
        
# Apply the function to create the 'Channel' column
winery['Channel'] = winery.apply(determine_channel, axis="columns")
# -

# #### Drop redundant sales channel columns

winery_clean = winery.drop(["EmailSales", "NewsletterSales", "TastingRoomSales", "WinemakerCallSales"], axis = "columns")
winery_clean.columns

# ### Creating Customer DataFrame

temp = winery_clean.drop(["Date", "Sales2008", "Sales2009", "Sales2010", "Orders2008", "Orders2009", "Orders2010" ,"Channel"], axis = "columns")
temp.head()

# +
customer = temp.groupby("CustomerID").agg({
    "OrderID": "count",
    "CustomerSegment": "first",
    "ZipCode": "first",
    "State": "first",
    "SaleAmount": "sum",
    "YearAcquired": "first",
    "EmailSubscr": "max",
    "NewsletterSubscr": "max",
    "WinemakerCallSubscr": "max",
    "Division": "first",
    "Region": "first"
}).reset_index()

customer.rename(columns = {'OrderID':'OrderVolume'}, inplace = True) 
customer.head()
# -

# ## Export csv

winery_clean.to_csv("../data/Winery_Data_Clean.csv", index = False)
customer.to_csv("../data/Winery_Customer.csv", index = False)
