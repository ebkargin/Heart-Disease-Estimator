
# 🫀 Kalp Hastalığı Tahmincisi

## 1 - Kullanılan Modeller

Proje süresince veri seti aşağıdaki üç farklı makine öğrenmesi algoritması ile eğitildi:

1. **XGBoost Classifier**
2. **Random Forest Classifier**
3. **Logistic Regression**

Modeller karşılaştırıldıktan sonra, **en iyi sonuçları veren XGBoost Classifier** tercih edildi.

---

## 2 - Performans Karşılaştırmaları

### ✅ Pozitif Sınıf (Kalp hastası bireyler)

| **MODEL**        | **ACC.** | **R_AUC** | **PREC.** | **REC.** | **F1**  |
|------------------|----------|-----------|-----------|----------|---------|
| **XGBOOST**      | 0.730    | **0.8006**| 0.76      | **0.69** | **0.72**|
| Random Forest    | **0.732**| 0.7993    | 0.76      | 0.68     | 0.71    |
| Logistic Regr.   | 0.712    | 0.7760    | 0.73      | 0.67     | 0.70    |

---

### ❌ Negatif Sınıf (Sağlıklı bireyler)

| **MODEL**        | **ACC.** | **R_AUC** | **PREC.** | **REC.** | **F1**  |
|------------------|----------|-----------|-----------|----------|---------|
| **XGBOOST**      | 0.730    | **0.8006**| 0.715     | **0.779**| **0.745**|
| Random Forest    | **0.732**| 0.7993    | **0.71**  | 0.786    | **0.745**|
| Logistic Regr.   | 0.712    | 0.7760    | 0.73      | 0.67     | 0.70    |

---

### 🔢 Karışıklık Matrisleri (Confusion Matrix)

- **XGBoost:**
  ```
  [[8181 2325]
   [3265 7229]]
  ```

- **Random Forest:**
  ```
  [[8266 2240]
   [3390 7104]]
  ```

- **Logistic Regression:**
  ```
  [[7908 2598]
   [3444 7050]]
  ```

---

## 3 - Değerlendirme ve Yorum

- **XGBoost**, ROC AUC skorunda **en yüksek değere** ulaşmıştır (**0.8006**) ve pozitif sınıf (hastalar) ayrımında **en başarılı modeldir**.
- **Random Forest** XGBoost’a **çok yakın sonuçlar** üretmiştir ancak **recall ve AUC skorlarında** geride kalmıştır.
- **Logistic Regression**, doğrusal sınıflandırma yapısından dolayı daha **düşük performans** göstermiştir çünkü veri seti doğrusal olmayan karmaşık ilişkiler içermektedir.
- Karışıklık matrisine göre **XGBoost**, pozitif sınıfı yani kalp hastası bireyleri tanıma konusunda diğer modellere göre daha başarılıdır.

---

## 4 - Sonuç

XGBoost modeli, **RandomizedSearchCV** ile hiperparametre optimizasyonu yapıldıktan sonra hem **genel doğruluk**, hem de **pozitif sınıf başarısı** açısından en iyi performansı göstermiştir.

Pozitif sınıfın (kalp hastası bireyler) **doğru tahmini kritik** olduğundan, bu projede **XGBoost tercih edilmiştir**. Random Forest modeli iyi bir alternatif olarak dursa da, hasta tahminindeki performans kriteri açısından geridedir.

---

## ⚙️ Kurulum

1. Python ortamınızı kurun (Python 3.10 veya üzeri önerilir):
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    ```

2. Gerekli kütüphaneleri yükleyin:
    ```bash
    pip install -r requirements.txt
    ```

3. `cardio_train.csv` dosyasını proje dizinine yerleştirin.

4. Programı başlatın:
    ```bash
    python main.py
    ```

---

> 📌 Not: `main.py` çalıştırıldığında eğer daha önce eğitilmiş bir model yoksa, sistem modeli otomatik olarak eğitir ve ardından kullanıcıdan veri alarak kalp hastalığı tahmininde bulunur.
