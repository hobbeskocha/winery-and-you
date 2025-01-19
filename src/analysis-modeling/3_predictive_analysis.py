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

# ## Import libraries and config Pandas display

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import joblib

from sklearn import set_config
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, auc

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
# -

pd.options.display.max_columns = 25
pd.options.display.max_rows = 100
set_config(transform_output = "pandas")

# ## Import datasets

winery = pd.read_csv("../../data/Winery_Data_Clean.csv", dtype={"Zipcode": object, })
winery.dtypes

# +
winery["Date"] = pd.to_datetime(winery["Date"], format="%Y-%m-%d")
winery = winery.astype({"CustomerID": object,
                        "OrderID": object})
categorical_columns = ["CustomerSegment", "State", "ZipCode", "Division", "Region", "Channel"]
winery[categorical_columns] = winery[categorical_columns].astype("category")

winery.dtypes
# -

customer = pd.read_csv("../../data/Winery_Customer.csv")
customer.dtypes

# +
customer = customer.astype({"CustomerID": object})
categorical_columns = ["CustomerSegment", "State", "ZipCode", "Division", "Region"]
customer[categorical_columns] = customer[categorical_columns].astype("category")

customer.dtypes
# -

# ## Predictive Models

# +
# TODO: cross-validation? cross_validate() for multiple metrics, KFold class + visualization of metrics
# TODO: RandomizedSearchCV - hyperparameter tuning

# TODO: Pipeline supports final prediction model, crossvalidation workflow, and processing steps -- make_pipeline()
# TODO: clustering with customer segment analysis for reclassification -- K-means, t-SNE, PCA?, NMF? + visualization of clusters/data points

#TODO: automate image export for all plots into /artifacts
# -

# ### Binary Classification

email_test = pd.DataFrame()
newsletter_test = pd.DataFrame()
winemaker_test = pd.DataFrame()

# #### Logistic Regression

# ##### Email Subscription

# +
# train test split for Email Subscription, stratified
X_train, X_test, y_train, y_test = train_test_split(
    customer.loc[:, ["OrderVolume", "CustomerSegment", "Division", "SaleAmount", "NewsletterSubscr", "WinemakerCallSubscr"]],
    customer.loc[:, "EmailSubscr"],
    test_size=0.2, random_state=0, stratify=customer.loc[:, "EmailSubscr"])

email_test["EmailSubscr"] = y_test

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
# -

log_email = sm.Logit(y_train, X_train_transform_const).fit()
log_email.summary()

significance = 0.05
sig_coeffs_email = [var for var, p in zip(log_email.pvalues.index, log_email.pvalues) if p < significance]
log_email.params[sig_coeffs_email]

# +
predictions_email = log_email.predict(X_test_transform_const)
email_test["prob_logit_email"] = predictions_email
predictions_email = np.round(predictions_email)
email_test["pred_logit_email"] = predictions_email

accuracy = accuracy_score(y_test, predictions_email)
precision = precision_score(y_test, predictions_email)
recall = recall_score(y_test, predictions_email)
f1score = f1_score(y_test, predictions_email)

print(f"Accuracy: {np.round(accuracy, 4)},\nPrecision: {np.round(precision, 4)},\nRecall: {np.round(recall, 4)},\nF1Score: {np.round(f1score, 4)}")
# -

cm = confusion_matrix(y_test, predictions_email)
plt.figure(figsize=(10, 6))
email_heatmap_plot = sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
email_heatmap_plot.set(xlabel = "Predicted", ylabel = "Actual", title = "Confusion Matrix of Email Subscr")
plt.show()

y_prob = log_email.predict(X_test_transform_const)
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve\nAccuracy: {:.2f}%'.format(
    accuracy * 100))
plt.legend(loc="lower right")
plt.show()

# Export artifacts

# +
email_scaler = StandardScaler()
email_scaler.fit(X_train[["SaleAmount", "OrderVolume"]])
email_sale_scale, email_sale_mean = email_scaler.scale_[0], email_scaler.mean_[0]
email_order_scale, email_order_mean = email_scaler.scale_[1], email_scaler.mean_[1]

print(email_sale_scale, email_sale_mean, email_order_scale, email_order_mean)
# -

joblib.dump(log_email, "../model-artifacts/log_email.pkl", compress=("zlib", 3))
with open("../model-artifacts/model-metrics.txt", "w") as f:
    f.write(f"Logit Email metrics:{np.round(accuracy, 4)};{email_sale_scale};{email_sale_mean};{email_order_scale};{email_order_mean},")

# ##### WinemakerCall Subscription

# +
# train test split for Winemaker Subscription, stratified
X_train, X_test, y_train, y_test = train_test_split(
    customer.loc[:, ["OrderVolume", "CustomerSegment", "Division", "SaleAmount", "NewsletterSubscr", "EmailSubscr"]],
    customer.loc[:, "WinemakerCallSubscr"],
    test_size=0.2, random_state=0, stratify=customer.loc[:, "WinemakerCallSubscr"])

winemaker_test["WinemakerCallSubscr"] = y_test

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
# -

log_winemaker = sm.Logit(y_train, X_train_transform_const).fit()
log_winemaker.summary()

significance = 0.05
sig_coeffs_winemaker = [var for var, p in zip(log_winemaker.pvalues.index, log_winemaker.pvalues) if p < significance]
log_winemaker.params[sig_coeffs_winemaker]

# +
predictions_winemaker = log_winemaker.predict(X_test_transform_const)
winemaker_test["prob_logit_winemaker"] = predictions_winemaker
predictions_winemaker = np.round(predictions_winemaker)
winemaker_test["pred_logit_winemaker"] = predictions_winemaker

accuracy = accuracy_score(y_test, predictions_winemaker)
precision = precision_score(y_test, predictions_winemaker)
recall = recall_score(y_test, predictions_winemaker)
f1score = f1_score(y_test, predictions_winemaker)

print(f"Accuracy: {np.round(accuracy, 4)},\nPrecision: {np.round(precision, 4)},\nRecall: {np.round(recall, 4)},\nF1Score: {np.round(f1score, 4)}")
# -

cm = confusion_matrix(y_test, predictions_winemaker)
plt.figure(figsize=(10, 6))
winemaker_heatmap_plot = sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
winemaker_heatmap_plot.set(xlabel = "Predicted", ylabel = "Actual", title = "Confusion Matrix of Winemaker Subscr")
plt.show()

y_prob = log_winemaker.predict(X_test_transform_const)
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve\nAccuracy: {:.2f}%'.format(
    accuracy * 100))
plt.legend(loc="lower right")
plt.show()

# Export Artifacts

# +
winemaker_scaler = StandardScaler()
winemaker_scaler.fit(X_train[["SaleAmount", "OrderVolume"]])
winemaker_sale_scale, winemaker_sale_mean = winemaker_scaler.scale_[0], winemaker_scaler.mean_[0]
winemaker_order_scale, winemaker_order_mean = winemaker_scaler.scale_[1], winemaker_scaler.mean_[1]

print(winemaker_sale_scale, winemaker_sale_mean, winemaker_order_scale, winemaker_order_mean)
# -

joblib.dump(log_winemaker, "../model-artifacts/log_winemaker.pkl", compress=("zlib", 3))
with open("../model-artifacts/model-metrics.txt", "a") as f:
    f.write(f"Logit Winemaker metrics:{np.round(accuracy, 4)};{winemaker_sale_scale};{winemaker_sale_mean};{winemaker_order_scale};{winemaker_order_mean},")

# ##### Newsletter Subscription

# +
# train test split for Newsletter Subscription, stratified
X_train, X_test, y_train, y_test = train_test_split(
    customer.loc[:, ["OrderVolume", "CustomerSegment", "Division", "SaleAmount", "WinemakerCallSubscr", "EmailSubscr"]],
    customer.loc[:, "NewsletterSubscr"],
    test_size=0.2, random_state=0, stratify=customer.loc[:, "NewsletterSubscr"])

newsletter_test["NewsletterSubscr"] = y_test

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
# -

log_newsletter = sm.Logit(y_train, X_train_transform_const).fit()
log_newsletter.summary()

significance = 0.05
sig_coeffs_newsletter = [var for var, p in zip(log_newsletter.pvalues.index, log_newsletter.pvalues) if p < significance]
log_newsletter.params[sig_coeffs_newsletter]

# +
predictions_newsletter = log_newsletter.predict(X_test_transform_const)
newsletter_test["prob_logit_newsletter"] = predictions_newsletter
predictions_newsletter = np.round(predictions_newsletter)
newsletter_test["pred_logit_newsletter"] = predictions_newsletter

accuracy = accuracy_score(y_test, predictions_newsletter)
precision = precision_score(y_test, predictions_newsletter)
recall = recall_score(y_test, predictions_newsletter)
f1score = f1_score(y_test, predictions_newsletter)

print(f"Accuracy: {np.round(accuracy, 4)},\nPrecision: {np.round(precision, 4)},\nRecall: {np.round(recall, 4)},\nF1Score: {np.round(f1score, 4)}")
# -

cm = confusion_matrix(y_test, predictions_newsletter)
plt.figure(figsize=(10, 6))
newsletter_heatmap_plot = sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
newsletter_heatmap_plot.set(xlabel = "Predicted", ylabel = "Actual", title = "Confusion Matrix of Newsletter Subscr")
plt.show()

y_prob = log_newsletter.predict(X_test_transform_const)
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve\nAccuracy: {:.2f}%'.format(
    accuracy * 100))
plt.legend(loc="lower right")
plt.show()

# Export Artifacts

# +
newsletter_scaler = StandardScaler()
newsletter_scaler.fit(X_train[["SaleAmount", "OrderVolume"]])
newsletter_sale_scale, newsletter_sale_mean = newsletter_scaler.scale_[0], newsletter_scaler.mean_[0]
newsletter_order_scale, newsletter_order_mean = newsletter_scaler.scale_[1], newsletter_scaler.mean_[1]

print(newsletter_sale_scale, newsletter_sale_mean, newsletter_order_scale, newsletter_order_mean)
# -

joblib.dump(log_newsletter, "../model-artifacts/log_newsletter.pkl", compress=("zlib", 3))
with open("../model-artifacts/model-metrics.txt", "a") as f:
    f.write(f"Logit Newsletter metrics:{np.round(accuracy, 4)};{newsletter_sale_scale};{newsletter_sale_mean};{newsletter_order_scale};{newsletter_order_mean},")

# #### Random Forest

# ##### Email Subscription

# train test split for Email Subscription, stratified
X_train, X_test, y_train, y_test = train_test_split(
    customer.loc[:, ["OrderVolume", "CustomerSegment", "Division", "SaleAmount", "NewsletterSubscr", "WinemakerCallSubscr"]],
    customer.loc[:, "EmailSubscr"],
    test_size=0.2, random_state=0, stratify=customer.loc[:, "EmailSubscr"])


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
print("Accuracy", np.round(accuracy, 4), "\n",
    "Precision", np.round(precision, 4), "\n",
    "Recall", np.round(recall, 4), "\n",
    "F1-Score", np.round(f1score, 4))

# +
email_75th_importance = np.percentile(feature_importances_email, 75)

for feature, importance in zip(X_train_transform.columns, feature_importances_email):
    if importance >= email_75th_importance:
        print(f"{feature}: {importance:.4f}")

# +
df_email_feat_import = pd.DataFrame(list(zip(X_train_transform.columns, feature_importances_email)), columns =['Feature', 'Importance'])
df_email_feat_import = df_email_feat_import.sort_values(by="Importance")

plt.figure(figsize=(8, 6))
email_feat_import_plot = sns.barplot(x='Importance', y='Feature', data=df_email_feat_import, hue='Feature', palette="colorblind")
email_feat_import_plot.set(xlabel = "Importance", ylabel = "Feature", title = "Feature Importance for the Email Random Forest Model")
plt.show()
# -

# Export Artifacts

# +
email_scaler = StandardScaler()
email_scaler.fit(X_train[["SaleAmount", "OrderVolume"]])
email_sale_scale, email_sale_mean = email_scaler.scale_[0], email_scaler.mean_[0]
email_order_scale, email_order_mean = email_scaler.scale_[1], email_scaler.mean_[1]

print(email_sale_scale, email_sale_mean, email_order_scale, email_order_mean)
# -

joblib.dump(classifier_email, "../model-artifacts/rf_email.pkl", compress=("zlib", 3))
with open("../model-artifacts/model-metrics.txt", "a") as f:
    f.write(f"RF Email metrics:{np.round(accuracy, 4)};{email_sale_scale};{email_sale_mean};{email_order_scale};{email_order_mean},")

# ##### WinemakerCall Subscription

# train test split for Winemaker Subscription, stratified
X_train, X_test, y_train, y_test = train_test_split(
    customer.loc[:, ["OrderVolume", "CustomerSegment", "Division", "SaleAmount", "NewsletterSubscr", "EmailSubscr"]],
    customer.loc[:, "WinemakerCallSubscr"],
    test_size=0.2, random_state=0, stratify=customer.loc[:, "WinemakerCallSubscr"])

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
print("Accuracy", np.round(accuracy, 4), "\n",
    "Precision", np.round(precision, 4), "\n",
    "Recall", np.round(recall, 4), "\n",
    "F1-Score", np.round(f1score, 4))

# +
winemaker_75th_importance = np.percentile(feature_importances_winemaker, 75)

for feature, importance in zip(X_train_transform.columns, feature_importances_winemaker):
    if importance >= winemaker_75th_importance:
        print(f"{feature}: {importance:.4f}")

# +
df_winemaker_feat_import = pd.DataFrame(list(zip(X_train_transform.columns, feature_importances_winemaker)), columns =['Feature', 'Importance'])
df_winemaker_feat_import = df_winemaker_feat_import.sort_values(by="Importance")

plt.figure(figsize=(8, 6))
winemaker_feat_import_plot = sns.barplot(x='Importance', y='Feature', data=df_winemaker_feat_import, hue='Feature', palette="colorblind")
winemaker_feat_import_plot.set(xlabel = "Importance", ylabel = "Feature", title = "Feature Importance for the Winemaker Random Forest Model")
plt.show()
# -

# Export Artifacts

# +
winemaker_scaler = StandardScaler()
winemaker_scaler.fit(X_train[["SaleAmount", "OrderVolume"]])
winemaker_sale_scale, winemaker_sale_mean = winemaker_scaler.scale_[0], winemaker_scaler.mean_[0]
winemaker_order_scale, winemaker_order_mean = winemaker_scaler.scale_[1], winemaker_scaler.mean_[1]

print(winemaker_sale_scale, winemaker_sale_mean, winemaker_order_scale, winemaker_order_mean)
# -

joblib.dump(classifier_winemaker, "../model-artifacts/rf_winemaker.pkl", compress=("zlib", 3))
with open("../model-artifacts/model-metrics.txt", "a") as f:
    f.write(f"RF Winemaker metrics:{np.round(accuracy, 4)};{winemaker_sale_scale};{winemaker_sale_mean};{winemaker_order_scale};{winemaker_order_mean},")

# ##### Newsletter Subscription

# train test split for Newsletter Subscription, stratified
X_train, X_test, y_train, y_test = train_test_split(
    customer.loc[:, ["OrderVolume", "CustomerSegment", "Division", "SaleAmount", "WinemakerCallSubscr", "EmailSubscr"]],
    customer.loc[:, "NewsletterSubscr"],
    test_size=0.2, random_state=0, stratify=customer.loc[:, "NewsletterSubscr"])

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
print("Accuracy", np.round(accuracy, 4), "\n",
    "Precision", np.round(precision, 4), "\n",
    "Recall", np.round(recall, 4), "\n",
    "F1-Score", np.round(f1score, 4))

# +
newsletter_75th_importance = np.percentile(feature_importances_newsletter, 75)

for feature, importance in zip(X_train_transform.columns, feature_importances_newsletter):
    if importance >= newsletter_75th_importance:
        print(f"{feature}: {importance:.4f}")

# +
df_newsletter_feat_import = pd.DataFrame(list(zip(X_train_transform.columns, feature_importances_newsletter)), columns =['Feature', 'Importance'])
df_newsletter_feat_import = df_newsletter_feat_import.sort_values(by="Importance")

plt.figure(figsize=(8, 6))
newsletter_feat_import_plot = sns.barplot(x='Importance', y='Feature', data=df_newsletter_feat_import, hue='Feature', palette="colorblind")
newsletter_feat_import_plot.set(xlabel = "Importance", ylabel = "Feature", title = "Feature Importance for the Newsletter Random Forest Model")
plt.show()
# -

# Export Artifacts

# +
newsletter_scaler = StandardScaler()
newsletter_scaler.fit(X_train[["SaleAmount", "OrderVolume"]])
newsletter_sale_scale, newsletter_sale_mean = newsletter_scaler.scale_[0], newsletter_scaler.mean_[0]
newsletter_order_scale, newsletter_order_mean = newsletter_scaler.scale_[1], newsletter_scaler.mean_[1]

print(newsletter_sale_scale, newsletter_sale_mean, newsletter_order_scale, newsletter_order_mean)
# -

joblib.dump(classifier_newsletter, "../model-artifacts/rf_newsletter.pkl", compress=("zlib", 3))
with open("../model-artifacts/model-metrics.txt", "a") as f:
    f.write(f"RF Newsletter metrics:{np.round(accuracy, 4)};{newsletter_sale_scale};{newsletter_sale_mean};{newsletter_order_scale};{newsletter_order_mean},")

# #### Calculate Lift

email_test.head(3)

newsletter_test.head(3)

winemaker_test.head(3)

# +
# avg response
avg_email = np.mean(email_test["EmailSubscr"])
avg_newsletter = np.mean(newsletter_test["NewsletterSubscr"])
avg_winemaker = np.mean(winemaker_test["WinemakerCallSubscr"])

# lift
email_test["lift_email"] = email_test["prob_logit_email"]/avg_email
newsletter_test["lift_newsletter"] = newsletter_test["prob_logit_newsletter"]/avg_newsletter
winemaker_test["lift_winemaker"] = winemaker_test["prob_logit_winemaker"]/avg_winemaker
# -

# #### Plot Marginal Response Rate

email_test_sorted = email_test.sort_values(by="lift_email", ascending=False)
marginal_email = sns.scatterplot(email_test_sorted, x = range(len(email_test_sorted)), y = "prob_logit_email", color = "black")
marginal_email.set(xlabel="Number of Prospects", ylabel="nth-best Response Rate", title="Email: Marginal Response Rate vs Number of Solicitations")
plt.show()

newsletter_test_sorted = newsletter_test.sort_values(by="lift_newsletter", ascending=False)
marginal_newsletter = sns.scatterplot(newsletter_test_sorted, x = range(len(newsletter_test_sorted)), y = "prob_logit_newsletter", color = "darkgreen")
marginal_newsletter.set(xlabel="Number of Prospects", ylabel="nth-best Response Rate", title="Newsletter: Marginal Response Rate vs Number of Solicitations")
plt.show()

winemaker_test_sorted = winemaker_test.sort_values(by="lift_winemaker", ascending=False)
marginal_winemaker = sns.scatterplot(winemaker_test_sorted, x = range(len(winemaker_test_sorted)), y = "prob_logit_winemaker", color = "red")
marginal_winemaker.set(xlabel="Number of Prospects", ylabel="nth-best Response Rate", title="Winemaker: Marginal Response Rate vs Number of Solicitations")
plt.show()

# #### Plot Number of Positive Reponses

# +
email_test_sorted["cumsum_email"] = email_test_sorted["prob_logit_email"].cumsum()
max_cumsum_email = max(email_test_sorted["cumsum_email"])

responses_email = sns.scatterplot(email_test_sorted, x = range(len(email_test_sorted)), y = "cumsum_email", color = "black")
responses_email.set(xlabel="Number of Customers", ylabel="Expected total number of responses", title="Email: Number of Positive Responses vs Number of Solicitations")
plt.show()

responses_email_prop = sns.scatterplot(email_test_sorted, 
                                       x = np.array(range(len(email_test_sorted)))/len(email_test_sorted), 
                                       y = email_test_sorted["cumsum_email"]/max_cumsum_email, 
                                       color = "black")
responses_email_prop.set(xlabel="Fraction of Customers", ylabel="Proportion of responses", title="Email: Proportion of Positive Responses vs Fraction of Solicitations")
plt.show()

# +
newsletter_test_sorted["cumsum_newsletter"] = newsletter_test_sorted["prob_logit_newsletter"].cumsum()
max_cumsum_newsletter = max(newsletter_test_sorted["cumsum_newsletter"])

responses_newsletter = sns.scatterplot(newsletter_test_sorted,
                                        x = range(len(newsletter_test_sorted)),
                                          y = "cumsum_newsletter",
                                            color = "darkgreen")
responses_newsletter.set(xlabel="Number of Customers", ylabel="Expected total number of responses", title="Newsletter: Number of Positive Responses vs Number of Solicitations")
plt.show()

responses_newsletter_prop = sns.scatterplot(newsletter_test_sorted, 
                                       x = np.array(range(len(newsletter_test_sorted)))/len(newsletter_test_sorted), 
                                       y = newsletter_test_sorted["cumsum_newsletter"]/max_cumsum_newsletter, 
                                       color = "darkgreen")
responses_newsletter_prop.set(xlabel="Fraction of Customers", ylabel="Proportion of responses", title="Newsletter: Proportion of Positive Responses vs Fraction of Solicitations")
plt.show()

# +
winemaker_test_sorted["cumsum_winemaker"] = winemaker_test_sorted["prob_logit_winemaker"].cumsum()
max_cumsum_winemaker = max(winemaker_test_sorted["cumsum_winemaker"])

responses_winemaker = sns.scatterplot(winemaker_test_sorted,
                                        x = range(len(winemaker_test_sorted)),
                                          y = "cumsum_winemaker",
                                            color = "darkgreen")
responses_winemaker.set(xlabel="Number of Customers", ylabel="Expected total number of responses", title="Winemaker: Number of Positive Responses vs Number of Solicitations")
plt.show()

responses_winemaker_prop = sns.scatterplot(winemaker_test_sorted, 
                                       x = np.array(range(len(winemaker_test_sorted)))/len(winemaker_test_sorted), 
                                       y = winemaker_test_sorted["cumsum_winemaker"]/max_cumsum_winemaker, 
                                       color = "darkgreen")
responses_winemaker_prop.set(xlabel="Fraction of Customers", ylabel="Proportion of responses", title="Winemaker: Proportion of Positive Responses vs Fraction of Solicitations")
plt.show()
# -

# ### Multinomial Classification

# create overall test-train from customer
cust_train_mc, cust_test_mc = train_test_split(customer, test_size=0.2, random_state=0, stratify=customer["CustomerSegment"])

# #### KNN

print(cust_train_mc["CustomerSegment"].value_counts(), "\n")
print(cust_test_mc["CustomerSegment"].value_counts())

# train and test subsets for Segment model
y_train = cust_train_mc.loc[:, "CustomerSegment"]
y_test = cust_test_mc.loc[:, "CustomerSegment"]
X_train = cust_train_mc.loc[:, ["OrderVolume", "Division", "SaleAmount", "NewsletterSubscr", "WinemakerCallSubscr", "EmailSubscr"]]
X_test = cust_test_mc.loc[:, ["OrderVolume", "Division", "SaleAmount", "NewsletterSubscr", "WinemakerCallSubscr", "EmailSubscr"]]

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
# -

k = int(np.round(np.sqrt(len(X_train_transform))))
knn = KNeighborsClassifier(n_neighbors = k).fit(X_train_transform, y_train)
knn_predictions = knn.predict(X_test_transform)
cm = confusion_matrix(y_test, knn_predictions)

plt.figure(figsize=(10, 8))
segment_heatmap_plot = sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
segment_heatmap_plot.set(xlabel = "Predicted", ylabel = "Actual", title = "Customer Segment KNN Confusion Matrix")
plt.show()

# +
accuracy = accuracy_score(y_test, knn_predictions)
precision = precision_score(y_test, knn_predictions, average='weighted')
recall = recall_score(y_test, knn_predictions, average='weighted')
f1 = f1_score(y_test, knn_predictions, average='weighted')

print(f"Accuracy: {np.round(accuracy, 4)},\nPrecision: {np.round(precision, 4)},\nRecall: {np.round(recall, 4)},\nF1Score: {np.round(f1score, 4)}")
# -

# #### Random Forest

# train and test subsets for Segment model
y_train = cust_train_mc.loc[:, "CustomerSegment"]
y_test = cust_test_mc.loc[:, "CustomerSegment"]
X_train = cust_train_mc.loc[:, ["OrderVolume", "Division", "SaleAmount", "NewsletterSubscr", "WinemakerCallSubscr", "EmailSubscr"]]
X_test = cust_test_mc.loc[:, ["OrderVolume", "Division", "SaleAmount", "NewsletterSubscr", "WinemakerCallSubscr", "EmailSubscr"]]

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

# +
classifier = RandomForestClassifier(n_estimators=50, random_state=0)
classifier.fit(X_train_transform, y_train)

predictions_rf = classifier.predict(X_test_transform)
feature_importances = classifier.feature_importances_

print("Feature Importance")
for feature, importance in zip(X_train_transform.columns, feature_importances):
    print(f"{feature}: {importance:.4f}")

accuracy = accuracy_score(y_test, predictions_rf)
precision = precision_score(y_test, predictions_rf, average="weighted")
recall = recall_score(y_test, predictions_rf, average="weighted")
f1score = f1_score(y_test, predictions_rf, average="weighted")

print("\nModel Metrics")
print("Accuracy", np.round(accuracy, 4), "\n",
    "Precision", np.round(precision, 4), "\n",
    "Recall", np.round(recall, 4), "\n",
    "F1-Score", np.round(f1score, 4))
# -

cm = confusion_matrix(y_test, predictions_rf)
plt.figure(figsize=(10, 6))
winemaker_heatmap_plot = sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
winemaker_heatmap_plot.set(xlabel = "Predicted", ylabel = "Actual", title = "Customer Segment Random Forest Confusion Matrix")
plt.show()
