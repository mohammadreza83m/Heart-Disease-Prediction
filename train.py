# train.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# 1. Load dataset
df = pd.read_csv("heart-disease.csv")

# 2. Feature engineering
df['risk'] = df['chol'] / df['age']
col = df.pop('risk')
df.insert(13, 'risk', col)

X = df.drop('target', axis=1)
y = df['target']

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Preprocessing
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# 5. Model (best hyperparameters from GridSearchCV)
clf = LogisticRegression(C=0.20433597178569418, solver='liblinear')

# 6. Final pipeline with SMOTE
pipe = ImbPipeline(steps=[
    ('preproc', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('clf', clf)
])

# 7. Train
pipe.fit(X_train, y_train)

# 8. Cross-validation scores
scoring = ['accuracy', 'precision', 'recall', 'f1']
cv_results = {s: cross_val_score(pipe, X, y, cv=5, scoring=s).mean() for s in scoring}
print("Cross-validation results:", cv_results)

# 9. Save trained model
joblib.dump(pipe, "logistic_model.pkl")
print("Model saved as logistic_model.pkl ✅")