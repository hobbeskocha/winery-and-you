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

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import set_config
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

import statsmodels.api as sm
# -

pd.options.display.max_columns = 25
pd.options.display.max_rows = 100
set_config(transform_output = "pandas")

# ## Import datasets

winery = pd.read_csv("../data/Winery_Data_Clean.csv", dtype={"Zipcode": object, })
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

# ## Predictive Models

# ### Binary Classification

# #### Logistic Regression

# create overall test-train from customer
cust_train, cust_test = train_test_split(customer, test_size=0.2, random_state=0)

# ##### Email Subscription

# train and test subsets for Email model
y_train = cust_train.loc[:, "EmailSubscr"]
y_test = cust_test.loc[:, "EmailSubscr"]
X_train = cust_train.loc[:, ["OrderVolume", "CustomerSegment", "Division", "SaleAmount", "NewsletterSubscr", "WinemakerCallSubscr"]]
X_test = cust_test.loc[:, ["OrderVolume", "CustomerSegment", "Division", "SaleAmount", "NewsletterSubscr", "WinemakerCallSubscr"]]

# +
number_features = list(X_train.select_dtypes(include=["int", "float"]).columns)
category_features = list(X_train.select_dtypes(include=["category", "bool"]).columns)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), number_features),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), category_features)
    ])

pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
pipeline.fit(X_train)
X_train_transform = pipeline.transform(X_train)
X_test_transform = pipeline.transform(X_test)

X_train_transform_const = sm.add_constant(X_train_transform)
X_test_transform_const = sm.add_constant(X_test_transform)

# +
# logistic_email = LogisticRegression(fit_intercept=False)
# logistic_email.fit(X_train_transform, y_train)

# logistic_email.coef_
# -

log_email = sm.Logit(y_train, X_train_transform_const).fit()
log_email.summary()

# +
predictions_email = log_email.predict(X_test_transform_const)
predictions_email = np.round(predictions_email)

accuracy = accuracy_score(y_test, predictions_email)
precision = precision_score(y_test, predictions_email)
recall = recall_score(y_test, predictions_email)
f1score = f1_score(y_test, predictions_email)

print("Acc", accuracy, "Prec", precision, "Rec", recall, "f1", f1score)
# -

cm = confusion_matrix(y_test, predictions_email)
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix of Email Subscr')
plt.show()

y_prob = log_email.predict(X_test_transform_const)
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve\nAccuracy: {:.2f}%'.format(
    accuracy * 100))
plt.legend(loc="lower right")
plt.show()

# ##### WinemakerCall Subscription

# train and test subsets for WinemakerCall model
y_train = cust_train.loc[:, "WinemakerCallSubscr"]
y_test = cust_test.loc[:, "WinemakerCallSubscr"]
X_train = cust_train.loc[:, ["OrderVolume", "CustomerSegment", "Division", "SaleAmount", "NewsletterSubscr", "EmailSubscr"]]
X_test = cust_test.loc[:, ["OrderVolume", "CustomerSegment", "Division", "SaleAmount", "NewsletterSubscr", "EmailSubscr"]]

# +
number_features = list(X_train.select_dtypes(include=["int", "float"]).columns)
category_features = list(X_train.select_dtypes(include=["category", "bool"]).columns)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), number_features),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), category_features)
    ])

pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
pipeline.fit(X_train)
X_train_transform = pipeline.transform(X_train)
X_test_transform = pipeline.transform(X_test)

X_train_transform_const = sm.add_constant(X_train_transform)
X_test_transform_const = sm.add_constant(X_test_transform)

# +
# logistic_winemaker = LogisticRegression()
# logistic_winemaker.fit(X_train, y_train)
# -

log_winemaker = sm.Logit(y_train, X_train_transform_const).fit()
log_winemaker.summary()

# +
predictions_winemaker = log_winemaker.predict(X_test_transform_const)
predictions_winemaker = np.round(predictions_winemaker)

accuracy = accuracy_score(y_test, predictions_winemaker)
precision = precision_score(y_test, predictions_winemaker)
recall = recall_score(y_test, predictions_winemaker)
f1score = f1_score(y_test, predictions_winemaker)

print("Acc", accuracy, "Prec", precision, "Rec", recall, "f1", f1score)
# -

cm = confusion_matrix(y_test, predictions_winemaker)
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix of Winemaker Subscr')
plt.show()

y_prob = log_winemaker.predict(X_test_transform_const)
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve\nAccuracy: {:.2f}%'.format(
    accuracy * 100))
plt.legend(loc="lower right")
plt.show()

# ##### Newsletter Subscription

# train and test subsets for Newsletter model
y_train = cust_train.loc[:, "NewsletterSubscr"]
y_test = cust_test.loc[:, "NewsletterSubscr"]
X_train = cust_train.loc[:, ["OrderVolume", "CustomerSegment", "Division", "SaleAmount", "WinemakerCallSubscr", "EmailSubscr"]]
X_test = cust_test.loc[:, ["OrderVolume", "CustomerSegment", "Division", "SaleAmount", "WinemakerCallSubscr", "EmailSubscr"]]

# +
number_features = list(X_train.select_dtypes(include=["int", "float"]).columns)
category_features = list(X_train.select_dtypes(include=["category", "bool"]).columns)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), number_features),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), category_features)
    ])

pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
pipeline.fit(X_train)
X_train_transform = pipeline.transform(X_train)
X_test_transform = pipeline.transform(X_test)

X_train_transform_const = sm.add_constant(X_train_transform)
X_test_transform_const = sm.add_constant(X_test_transform)

# +
# logistic_newsletter = LogisticRegression()
# logistic_newsletter.fit(X_train, y_train)
# -

log_newsletter = sm.Logit(y_train, X_train_transform_const).fit()
log_newsletter.summary()

# +
predictions_newsletter = log_newsletter.predict(X_test_transform_const)
predictions_newsletter = np.round(predictions_newsletter)

accuracy = accuracy_score(y_test, predictions_newsletter)
precision = precision_score(y_test, predictions_newsletter)
recall = recall_score(y_test, predictions_newsletter)
f1score = f1_score(y_test, predictions_newsletter)

print("Acc", accuracy, "Prec", precision, "Rec", recall, "f1", f1score)
# -

cm = confusion_matrix(y_test, predictions_newsletter)
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix of Winemaker Subscr')
plt.show()

y_prob = log_newsletter.predict(X_test_transform_const)
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve\nAccuracy: {:.2f}%'.format(
    accuracy * 100))
plt.legend(loc="lower right")
plt.show()

# #### Random Forest

# ##### Email Subscription

y = customer.loc[:, "EmailSubscr"]
X = customer.loc[:, ["OrderVolume", "CustomerSegment", "Division", "SaleAmount", "NewsletterSubscr", "WinemakerCallSubscr"]]

# +
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

number_features = list(X.select_dtypes(include=["int", "float"]).columns)
category_features = list(X.select_dtypes(include=["category", "bool"]).columns)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), number_features),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), category_features)
    ])

pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
pipeline.fit(X_train)
X_train_transform = pipeline.transform(X_train)
X_test_transform = pipeline.transform(X_test)


# +
classifier_email = RandomForestClassifier(random_state=0)
classifier_email.fit(X_train_transform, y_train)

predictions_email = classifier_email.predict(X_test_transform)
predictions_email = np.round(predictions_email)
feature_importances_email = classifier_email.feature_importances_

print("Feature Importance")
for feature, importance in zip(X_train_transform.columns, feature_importances_email):
    print(f"{feature}: {importance:.4f}")

accuracy = accuracy_score(y_test, predictions_email)
precision = precision_score(y_test, predictions_email)
recall = recall_score(y_test, predictions_email)
f1score = f1_score(y_test, predictions_email)

print("\nModel Metrics")
print("Accuracy", accuracy, "\n",
    "Precision", precision, "\n",
    "Recall", recall, "\n",
    "F1-Score", f1score)
# -

# ##### WinemakerCall Subscription

y = customer.loc[:, "WinemakerCallSubscr"]
X = customer.loc[:, ["OrderVolume", "CustomerSegment", "Division", "SaleAmount", "NewsletterSubscr", "EmailSubscr"]]

# +
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

number_features = list(X.select_dtypes(include=["int", "float"]).columns)
category_features = list(X.select_dtypes(include=["category", "bool"]).columns)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), number_features),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), category_features)
    ])

pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
pipeline.fit(X_train)
X_train_transform = pipeline.transform(X_train)
X_test_transform = pipeline.transform(X_test)


# +
classifier_winemaker = RandomForestClassifier(random_state=0)
classifier_winemaker.fit(X_train_transform, y_train)

predictions_winemaker = classifier_winemaker.predict(X_test_transform)
predictions_winemaker = np.round(predictions_winemaker)
feature_importances_winemaker = classifier_winemaker.feature_importances_

print("Feature Importance")
for feature, importance in zip(X_train_transform.columns, feature_importances_winemaker):
    print(f"{feature}: {importance:.4f}")

accuracy = accuracy_score(y_test, predictions_winemaker)
precision = precision_score(y_test, predictions_winemaker)
recall = recall_score(y_test, predictions_winemaker)
f1score = f1_score(y_test, predictions_winemaker)

print("\nModel Metrics")
print("Accuracy", accuracy, "\n",
    "Precision", precision, "\n",
    "Recall", recall, "\n",
    "F1-Score", f1score)
# -

# ##### Newsletter Subscription

y = customer.loc[:, "NewsletterSubscr"]
X = customer.loc[:, ["OrderVolume", "CustomerSegment", "Division", "SaleAmount", "WinemakerCallSubscr", "EmailSubscr"]]

# +
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

number_features = list(X.select_dtypes(include=["int", "float"]).columns)
category_features = list(X.select_dtypes(include=["category", "bool"]).columns)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), number_features),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), category_features)
    ])

pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
pipeline.fit(X_train)
X_train_transform = pipeline.transform(X_train)
X_test_transform = pipeline.transform(X_test)


# +
classifier_newsletter = RandomForestClassifier(random_state=0)
classifier_newsletter.fit(X_train_transform, y_train)

predictions_newsletter = classifier_newsletter.predict(X_test_transform)
predictions_newsletter = np.round(predictions_newsletter)
feature_importances_newsletter = classifier_newsletter.feature_importances_

print("Feature Importance")
for feature, importance in zip(X_train_transform.columns, feature_importances_newsletter):
    print(f"{feature}: {importance:.4f}")

accuracy = accuracy_score(y_test, predictions_newsletter)
precision = precision_score(y_test, predictions_newsletter)
recall = recall_score(y_test, predictions_newsletter)
f1score = f1_score(y_test, predictions_newsletter)

print("\nModel Metrics")
print("Accuracy", accuracy, "\n",
    "Precision", precision, "\n",
    "Recall", recall, "\n",
    "F1-Score", f1score)
# -

# #### Calculate Lift, Marginal Response Rate, Number of Positive Reponses

# probability, prediction, avg response, lift, marginal response rate, number of positive responses for email, winemaker, newsletter


# ### Multinomial Classification

# #### KNN

customer["CustomerSegment"].value_counts()

# +
y = customer.loc[:, "CustomerSegment"]
X = customer.loc[:, ["OrderVolume", "Region", "SaleAmount", "NewsletterSubscr", "WinemakerCallSubscr", "EmailSubscr"]]

# TODO: use dummies
label_encoder = LabelEncoder()
X["Region"] = label_encoder.fit_transform(X["Region"])

# +
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state = 0)

k = int(np.round(np.sqrt(len(X_train))))
knn = KNeighborsClassifier(n_neighbors = k).fit(X_train, y_train)

knn_predictions = knn.predict(X_test)
cm = confusion_matrix(y_test, knn_predictions)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

print(k)

# +
accuracy = accuracy_score(y_test, knn_predictions)
precision = precision_score(y_test, knn_predictions, average='weighted')
recall = recall_score(y_test, knn_predictions, average='weighted')
f1 = f1_score(y_test, knn_predictions, average='weighted')

print("Acc", accuracy, "Prec", precision, "Rec", recall, "f1", f1score)
# -

# #### Random Forest

# +
y = customer.loc[:, "CustomerSegment"]
X = customer.loc[:, ["OrderVolume", "Region", "SaleAmount", "NewsletterSubscr", "WinemakerCallSubscr", "EmailSubscr"]]

# TODO: use dummies
label_encoder = LabelEncoder()
X["Region"] = label_encoder.fit_transform(X["Region"])
y = label_encoder.fit_transform(y)
# -

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state = 0)

# +
regressor = RandomForestClassifier(n_estimators=50, random_state=0)
regressor.fit(X_train, y_train)

predictions = regressor.predict(X_test)

feature_importances = regressor.feature_importances_
for feature, importance in zip(["OrderVolume", "Region", "SaleAmount", "NewsletterSubscr", "WinemakerCallSubscr", "EmailSubscr"]
                               , feature_importances):
    print(f"{feature}: {importance:.4f}")

accuracy = accuracy_score(y_test, np.round(predictions))
precision = precision_score(y_test, np.round(predictions), average="weighted")
recall = recall_score(y_test, np.round(predictions), average="weighted")
f1score = f1_score(y_test, np.round(predictions), average="weighted")

print()
print("Acc", accuracy, "Prec", precision, "Rec", recall, "f1", f1score)
