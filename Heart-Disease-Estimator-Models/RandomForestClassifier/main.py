import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

df = pd.read_csv("cardio_train.csv", delimiter=';')

df = df.rename(columns={
    "id": "id",
    "age": "yas",
    "gender": "cinsiyet",
    "height": "boy_cm",
    "weight": "kilo_kg",
    "ap_hi": "buyuk_tansiyon",
    "ap_lo": "kucuk_tansiyon",
    "cholesterol": "kolesterol",
    "gluc": "glukoz",
    "smoke": "sigara_iciyor",
    "alco": "alkol_aliyor",
    "active": "fiziksel_aktif",
    "cardio": "kalp_hastaligi"
})

df['yas'] = (df['yas'] / 365.25).astype(int)
df['bmi'] = df['kilo_kg'] / ((df['boy_cm'] / 100) ** 2)
df['tansiyon_farki'] = df['buyuk_tansiyon'] - df['kucuk_tansiyon']
df.drop(columns='id', inplace=True)

X = df.drop('kalp_hastaligi', axis=1)
y = df['kalp_hastaligi']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

rf = RandomForestClassifier(random_state=42)
rs_cv = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=20,
    cv=3,
    scoring='roc_auc',
    verbose=1,
    n_jobs=-1,
    random_state=42
)
rs_cv.fit(x_train, y_train)
best_rf = rs_cv.best_estimator_
y_pred = best_rf.predict(x_test)
y_prob = best_rf.predict_proba(x_test)[:, 1]
print(" En iyi parametreler:")
print(rs_cv.best_params_)
print("\n Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred))
print("Karışıklık Matrisi:")
print(confusion_matrix(y_test, y_pred))
print("ROC AUC Skoru:")
print(roc_auc_score(y_test, y_prob))
print("Test Skoru:", best_rf.score(x_test, y_test))
print("Train Skoru:", best_rf.score(x_train, y_train))
