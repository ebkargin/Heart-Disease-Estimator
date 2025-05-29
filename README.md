# Kalp Hastalığı Riski Tahmin Projesi

## Proje Özeti

Bu projede, kalp hastalığı riskini tahmin etmeye yönelik çeşitli makine öğrenmesi modelleri eğitilmiş ve karşılaştırılmıştır. Kullanılan veri seti, bireylerin çeşitli tıbbi ve demografik özelliklerini içermektedir. Amaç, özellikle kalp hastası olan bireyleri (pozitif sınıf) doğru şekilde tespit etmektir; çünkü bu sınıfın doğru sınıflandırılması, sağlık açısından kritik önem taşımaktadır.

## Proje Detayları

### 1. Kullanılan Modeller:

1. XGBoost Classifier  
2. Random Forest Classifier  
3. Logistic Regression  
4. KNN Classifier  

Proje süresince veri seti bu modellere göre eğitilmiş olup aralarından en iyi sonuçları veren XGBoost Classifier seçilmiştir.

### 2. Performans Karşılaştırmaları

#### Pozitif Sınıf:

| MODEL           | ACC. | R_AUC | PREC. | REC  | F1   |
|------------------|------|--------|--------|------|------|
| XGBOOST         | 0.73 | 0.80   | 0.76   | 0.69 | 0.72 |
| Random Forest   | 0.732| 0.798  | 0.76   | 0.68 | 0.71 |
| Logistic Regr.  | 0.72 | 0.784  | 0.74   | 0.69 | 0.72 |
| KNN Class.      | 0.55 | 0.568  | 0.55   | 0.537| 0.544|

#### Negatif Sınıf:

| MODEL           | ACC. | R_AUC | PREC. | REC  | F1   |
|------------------|------|--------|--------|------|------|
| XGBOOST         | 0.73 | 0.80   | 0.715  | 0.779| 0.745|
| Random Forest   | 0.732| 0.798  | 0.71   | 0.786| 0.75 |
| Logistic Regr.  | 0.72 | 0.784  | 0.71   | 0.76 | 0.73 |
| KNN Class.      | 0.55 | 0.568  | 0.549  | 0.56 | 0.55 |

### Karışıklık Matrisleri:

**XGBoost:**  
[[8181 2325]  
 [3265 7229]]

**Logistic Regression:**  
[[5297 1670]  
 [2190 4843]]

**Random Forest Classifier:**  
[[8266 2240]  
 [3390 7104]]

**KNN Classifier:**  
[[3008 2296]  
 [2384 2812]]

### 3. Değerlendirme ve Yorum:

- XGBoost, ROC AUC skorunda en yüksek değere sahip olan modeldir (0.8006) ve modelin pozitif sınıfı ayırt etme başarısında en iyi performansı gösterdi.
- Random Forest modeli XGBoost’a yakın sonuçlar üretti ancak recall ve AUC skorlarında geri kaldı.
- Logistic Regression, doğrusal sınırlamalardan dolayı diğer modellere göre daha düşük performans gösterdi çünkü elimizdeki veri seti karmaşık ilişkilerden oluşmakta.
- Karışıklık Matrisi sonuçları, XGBoost’un pozitif sınıfı yani kalp hastalarını tahminlemekte daha başarılı olduğunu göstermiştir.

### 4. Sonuç:

XGBoost modeli RandomizedSearchCV ile hiperparametre optimizasyonu yapıldıktan sonra hem genel doğruluk hem de pozitif sınıf başarısı açısından en iyi performansı göstermiştir. Pozitif sınıfın yani kalp hastası bireylerin tahmini kritik olduğundan, bu projede XGBoost tercih edilmiştir. Random Forest Classifier her ne kadar hiperparametre optimizasyonu yapıldığında çok ufak bir farkla XGBoost’a göre daha yüksek accuracy’e sahip olsa da pozitif sınıfın tahmininde XGBoost’a göre geri kalmıştır.