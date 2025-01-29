from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import statsmodels.api as sm

def pipeline_transform(X_train, X_test, model_type):
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

    if model_type == "Logit":
        X_train_transform = sm.add_constant(X_train_transform)
        X_test_transform = sm.add_constant(X_test_transform)

    return X_train_transform, X_test_transform