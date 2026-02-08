import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from utils import FeatureEngineer, IQRClipper
import joblib

# 1. Load your dataset
df = pd.read_csv("water_potability.csv")

X = df.drop("Potability", axis=1)
y = df["Potability"]

# 2. Preprocessor
num_features = X.columns

preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ("iqr", IQRClipper()),
        ("feat", FeatureEngineer()),
        ('scaler', StandardScaler())
    ]), num_features)
])

# 3. Base classifiers
clf_rf = RandomForestClassifier(n_estimators=271, max_depth=None, min_samples_split=3, random_state=42)
clf_gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
clf_svm = SVC(probability=True, random_state=42)


# 4. Stacking classifier
stacking_clf = StackingClassifier(
    estimators=[
        ('rf', clf_rf),
        ('gb', clf_gb),
        ('svm', clf_svm)
    ],
    final_estimator=LogisticRegression(max_iter=1000),
    passthrough=False
)


# 5. Full pipeline
stacking_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', stacking_clf)
])


# 6. Fit the pipeline
stacking_pipeline.fit(X, y)


# 7. Save the pipeline
joblib.dump(stacking_pipeline, "stacking_pipeline_water.joblib")