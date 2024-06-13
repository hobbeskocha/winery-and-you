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

import sklearn 
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import r2_score, roc_curve, auc
# -

pd.options.display.max_columns = 25
pd.options.display.max_rows = 100

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

# ##### Email Subscription

y = customer.loc[:, "EmailSubscr"]
X = customer.loc[:, ["OrderVolume", "CustomerSegment", "Region", "SaleAmount", "NewsletterSubscr", "WinemakerCallSubscr"]]

# +
### TODO: introduce dummies rather than label encoding

label_encoder = LabelEncoder()
x_cat = X.select_dtypes(include=['object', 'bool']).apply(label_encoder.fit_transform)
x_num = X.select_dtypes(exclude=['object', 'bool'])

X_concat = pd.concat([pd.DataFrame(x_num), pd.DataFrame(x_cat)], axis='columns')

# +
X_train, X_test, y_train, y_test = train_test_split(X_concat, y, test_size=0.2, random_state=0)

scaler = StandardScaler()
X_train["OrderVolume"] = scaler.fit_transform(X_train[["OrderVolume"]])
X_train["SaleAmount"] = scaler.fit_transform(X_train[["SaleAmount"]])

# TODO: fit_transform on train data => transform on test
X_test["OrderVolume"] = scaler.fit_transform(X_test[["OrderVolume"]])
X_test["SaleAmount"] = scaler.fit_transform(X_test[["SaleAmount"]])
# -

logistic_email = LogisticRegression()
logistic_email.fit(X_train, y_train)

# +
predictions = logistic_email.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1score = f1_score(y_test, predictions)

print("Acc", accuracy, "Prec", precision, "Rec", recall, "f1", f1score)
# -

cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix of Email Subscr')
plt.show()

y_prob = logistic_email.predict_proba(X_test)[:, 1]
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

# ##### WinemakerCall Susbcription

y = customer.loc[:, "WinemakerCallSubscr"]
X = customer.loc[:, ["OrderVolume", "CustomerSegment", "Region", "SaleAmount", "NewsletterSubscr", "EmailSubscr"]]

# +
### TODO: introduce dummies rather than label encoding

label_encoder = LabelEncoder()
x_cat = X.select_dtypes(include=['object', 'bool']).apply(label_encoder.fit_transform)
x_num = X.select_dtypes(exclude=['object', 'bool'])

X_concat = pd.concat([pd.DataFrame(x_num), pd.DataFrame(x_cat)], axis='columns')

# +
X_train, X_test, y_train, y_test = train_test_split(X_concat, y, test_size=0.2, random_state=0)

scaler = StandardScaler()
X_train["OrderVolume"] = scaler.fit_transform(X_train[["OrderVolume"]])
X_train["SaleAmount"] = scaler.fit_transform(X_train[["SaleAmount"]])

# TODO: fit_transform on train data => transform on test
X_test["OrderVolume"] = scaler.fit_transform(X_test[["OrderVolume"]])
X_test["SaleAmount"] = scaler.fit_transform(X_test[["SaleAmount"]])
# -

logistic_winemaker = LogisticRegression()
logistic_winemaker.fit(X_train, y_train)

# +
predictions = logistic_winemaker.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1score = f1_score(y_test, predictions)

print("Acc", accuracy, "Prec", precision, "Rec", recall, "f1", f1score)
# -

cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix of Winemaker Subscr')
plt.show()

y_prob = logistic_winemaker.predict_proba(X_test)[:, 1]
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

y = customer.loc[:, "NewsletterSubscr"]
X = customer.loc[:, ["OrderVolume", "CustomerSegment", "Region", "SaleAmount", "WinemakerCallSubscr", "EmailSubscr"]]

# +
### TODO: introduce dummies rather than label encoding

label_encoder = LabelEncoder()
x_cat = X.select_dtypes(include=['object', 'bool']).apply(label_encoder.fit_transform)
x_num = X.select_dtypes(exclude=['object', 'bool'])

X_concat = pd.concat([pd.DataFrame(x_num), pd.DataFrame(x_cat)], axis='columns')

# +
X_train, X_test, y_train, y_test = train_test_split(X_concat, y, test_size=0.2, random_state=0)

scaler = StandardScaler()
X_train["OrderVolume"] = scaler.fit_transform(X_train[["OrderVolume"]])
X_train["SaleAmount"] = scaler.fit_transform(X_train[["SaleAmount"]])

# TODO: fit_transform on train data => transform on test
X_test["OrderVolume"] = scaler.fit_transform(X_test[["OrderVolume"]])
X_test["SaleAmount"] = scaler.fit_transform(X_test[["SaleAmount"]])
# -

logistic_newsletter = LogisticRegression()
logistic_newsletter.fit(X_train, y_train)

# +
predictions = logistic_newsletter.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1score = f1_score(y_test, predictions)

print("Acc", accuracy, "Prec", precision, "Rec", recall, "f1", f1score)
# -

cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix of Winemaker Subscr')
plt.show()

y_prob = logistic_newsletter.predict_proba(X_test)[:, 1]
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

X = pd.get_dummies(X, columns=["CustomerSegment", "Division"], drop_first=True)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# +
classifier_email = RandomForestClassifier(random_state=0)
classifier_email.fit(X_train, Y_train)

predictions_email = classifier_email.predict(X_test)
feature_importances_email = classifier_email.feature_importances_

print("Feature Importance")
for feature, importance in zip(X.columns, feature_importances_email):
    print(f"{feature}: {importance:.4f}")

accuracy = accuracy_score(Y_test, np.round(predictions_email))
precision = precision_score(Y_test, np.round(predictions_email))
recall = recall_score(Y_test, np.round(predictions_email))
f1score = f1_score(Y_test, np.round(predictions_email))

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
X = pd.get_dummies(X, columns=["CustomerSegment", "Division"], drop_first=True)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# +
classifier_winemaker = RandomForestClassifier(random_state=0)
classifier_winemaker.fit(X_train, Y_train)

predictions_winemaker = classifier_winemaker.predict(X_test)
feature_importances_winemaker = classifier_winemaker.feature_importances_

print("Feature Importance")
for feature, importance in zip(X.columns, feature_importances_winemaker):
    print(f"{feature}: {importance:.4f}")

accuracy = accuracy_score(Y_test, np.round(predictions_winemaker))
precision = precision_score(Y_test, np.round(predictions_winemaker))
recall = recall_score(Y_test, np.round(predictions_winemaker))
f1score = f1_score(Y_test, np.round(predictions_winemaker))

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
X = pd.get_dummies(X, columns=["CustomerSegment", "Division"], drop_first=True)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# +
classifier_newsletter = RandomForestClassifier(random_state=0)
classifier_newsletter.fit(X_train, Y_train)

predictions_newsletter = classifier_newsletter.predict(X_test)
feature_importances_newsletter = classifier_newsletter.feature_importances_

print("Feature Importance")
for feature, importance in zip(X.columns, feature_importances_newsletter):
    print(f"{feature}: {importance:.4f}")

accuracy = accuracy_score(Y_test, np.round(predictions_newsletter))
precision = precision_score(Y_test, np.round(predictions_newsletter))
recall = recall_score(Y_test, np.round(predictions_newsletter))
f1score = f1_score(Y_test, np.round(predictions_newsletter))

print("\nModel Metrics")
print("Accuracy", accuracy, "\n",
    "Precision", precision, "\n",
    "Recall", recall, "\n",
    "F1-Score", f1score)
# -

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
