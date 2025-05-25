import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import joblib


def int_input(prompt,min_value,max_value):
    while True:
        try:
            deger = int(input(prompt))
            if min_value <= deger <= max_value:
                return deger
            else:
                print(f"{min_value} ile {max_value} arasında bir değer giriniz!")
        except ValueError:
            print("Hatalı giriş! Lütfen sadece sayı giriniz.")

def bool_input(prompt):
    while True:
        try:
            deger = int(input(prompt))
            if deger in (0,1):
                return deger
            else:
                print("Lütfen sadece 0 (Hayır) veya 1 (Evet) giriniz!!!")
        except ValueError:
            print("Hatalı giriş 0 veya 1 giriniz!!!")

def float_input(prompt, min_value, max_value):
    while True:
        try:
            deger = float(input(prompt))
            if min_value <= deger <= max_value:
                return deger
            else:
                print(f"{min_value} ile {max_value} arasında bir değer giriniz!")
        except ValueError:
            print("Hatalı giriş! Lütfen sadece sayı giriniz.")

def gender_input(prompt):
    while True:
        try:
            deger = int(input(prompt))
            if deger in (1,2):
                return deger
            else:
                print("Lütfen sadece 1 (Kadın) veya 2 (Erkek) giriniz!!!")
        except ValueError:
            print("Hatalı giriş lütfen sadece 1 (Kadın) veya 2 (Erkek) giriniz!!!")

def train_save():
    df = pd.read_csv('cardio_train.csv', delimiter=';')
    df = df.rename(columns={ #Sütunlar Türkçeleştirildi.
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
    df['yas'] = (df['yas'] / 365.25).astype(int) #Gün cinsinden verilen yaş yıla çevrilir.
    df.drop(columns='id', inplace=True) #id kalp rahatsızlığı tahmininde gereksiz bir nitelik olduğu için attım
    df['bmi'] = df['kilo_kg'] / ((df['boy_cm'] / 100) ** 2) #Model performansını arttırmak amacıyla vücut kitle endeksi ekledim.
    df['tansiyon_farki'] = df['buyuk_tansiyon'] - df['kucuk_tansiyon'] #Model performansını arttırmak amacıyla yapıldı.

    x = df.drop(columns='kalp_hastaligi') #Kalp hastalığı hariç tüm öznitelikler.
    y = df['kalp_hastaligi'] #Kalp hastalığı tahminlenmek istenen nitelik.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=42) #Veriseti eğitim amaçla train ve deneme amaçlı test olmak üzere ikiye bölündü.

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.6,
        colsample_bytree=0.6,
        min_child_weight=1,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(x_train, y_train)
    joblib.dump(model, 'heart_disease_model.pkl') #Model eğitimini her çalıştırmada tekrar tekrar yapmamak amacıyla joblib kütüphanesinden dump metoduyla diske kaydettim.

def program():
    model = joblib.load('heart_disease_model.pkl') #Eğitilmiş modeli diskten çektim.
    print("\nLütfen aşağıdaki bilgileri giriniz:")
    yas = int_input("Yaş: ", 1, 120)
    cinsiyet = gender_input("Cinsiyetinizi tuşlayın (1=Kadın, 2=Erkek): ")
    boy = int_input("Boy (cm): ", 100, 250)
    kilo = float_input("Kilo (kg): ", 30.0, 400)
    buyuk_tansiyon = int_input("Büyük Tansiyon (Normal Değerler: 90 – 120): ", 50, 200)
    kucuk_tansiyon = int_input("Küçük Tansiyon (Normal Değerler: 60 – 80): ", 30, 130)
    kolesterol = int_input("Kolesterol (1:Normal < 200 mg/dL, 2:Yüksek 200–239 mg/dL, 3:Çok Yüksek ≥ 240 mg/dL): ", 1, 3)
    glukoz = int_input("Açlık Kan Şekeri (1:Normal 70 – 99 mg/dL, 2:Yüksek 	100 – 125 mg/dL, 3:Çok Yüksek ≥ 126 mg/dL): ", 1, 3)
    sigara =bool_input("Sigara içiyor musunuz (0:Hayır, 1:Evet): ")
    alkol = bool_input("Alkol tüketiyor musunuz (0:Hayır, 1:Evet): ")
    aktif =bool_input("Fiziksel Aktivite yapıyor musunuz(0:Hayır, 1:Evet): ")
    bmi = kilo / ((boy / 100) ** 2)
    tansiyon_farki = buyuk_tansiyon - kucuk_tansiyon
    veri = pd.DataFrame([[ #Veri adında boş bir dataframe oluşturur. İçerisine kullanıcıdan alınan verileri karşılık gelen sütunlara atar.
        yas, cinsiyet, boy, kilo, buyuk_tansiyon, kucuk_tansiyon, kolesterol,
        glukoz, sigara, alkol, aktif, bmi, tansiyon_farki
    ]], columns=[
        'yas', 'cinsiyet', 'boy_cm', 'kilo_kg', 'buyuk_tansiyon',
        'kucuk_tansiyon', 'kolesterol', 'glukoz', 'sigara_iciyor',
        'alkol_aliyor', 'fiziksel_aktif', 'bmi', 'tansiyon_farki'
    ])
    olasilik = model.predict_proba(veri)[0][1] #Hasta olma olasılığını döndürür.
    tahmin = model.predict(veri)[0] #0 veya 1 döndürür. 0 sağlklı, 1 hasta.

    print(f"\nKalp hastalığına yakalanma olasılığınız: %{olasilik * 100:.2f}")
    if tahmin == 1:
        print("Model tahmini: Riskli (Kalp hastalığına yakalanabilirsiniz. Lütfen daha sağlıklı bir hayat yaşamaya gayret gösterin.)")
    else:
        print("Model tahmini: Sağlıklı (Kalp hastalığına yakalanma riskiniz düşük.)")

if __name__ == "__main__":
    import os
    if not os.path.exists("heart_disease_model.pkl"): #heart_disease_model.pkl kayıtlı değilse train_save metodunu çağırır ve kaydeder.
        train_save()
    program() #Daha sonrasında program metodu çağrılır ve hastadan veriler alınarak tahminleme işlemi yapar. Import edilerek çağrılırsa bu kısım çalışmayacaktır.