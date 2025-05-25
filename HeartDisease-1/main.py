import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

df = pd.read_csv('heart_2020_cleaned.csv')
age_mapping = {
    "18-24": 21, "25-29": 27, "30-34": 32, "35-39": 37,
    "40-44": 42, "45-49": 47, "50-54": 52, "55-59": 57,
    "60-64": 62, "65-69": 67, "70-74": 72, "75-79": 77,
    "80 or older": 85
}
df['Age'] = df['AgeCategory'].map(age_mapping)
df = df.drop('AgeCategory', axis=1)
df = pd.get_dummies(df, drop_first=True)
y = df['HeartDisease_Yes']
x = df.drop('HeartDisease_Yes', axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
smote = SMOTE(random_state=42)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 4, 10],
    'class_weight': ['balanced']
}
rfc = RandomForestClassifier(random_state=42)
random_search = RandomizedSearchCV(
    estimator=rfc,
    param_distributions=param_dist,
    n_iter=20,               # Sadece 20 kombinasyon denenir
    scoring='f1',
    cv=3,
    verbose=2,
    n_jobs=-1,
    random_state=42
)
random_search.fit(x_train_res, y_train_res)
best_model = random_search.best_estimator_
print("🎯 En iyi parametreler:", random_search.best_params_)
y_pred = best_model.predict(x_test)
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))
print("📉 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\n✅ Train Score:", best_model.score(x_train_res, y_train_res))
print("✅ Test Score :", best_model.score(x_test, y_test))
