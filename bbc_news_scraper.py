import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords

# Gerekli NLTK verilerini indir
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

# 1. BBC News'ten veri çekme (Örnek: Teknoloji haberleri)
URL = "https://www.bbc.com/news/technology"
headers = {"User-Agent": "Mozilla/5.0"}
response = requests.get(URL, headers=headers)
response.raise_for_status()  # Hata kontrolü
soup = BeautifulSoup(response.text, "html.parser")

# 2. Haber başlıklarını alma
headlines = [headline.text.strip() for headline in soup.find_all("h3")]

# 3. Veriyi temizleme
def clean_text(text):
    text = text.lower()  # Küçük harfe çevirme
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Özel karakterleri kaldırma
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Stopwords kaldırma
    return text

cleaned_headlines = [clean_text(headline) for headline in headlines if headline]

df = pd.DataFrame(cleaned_headlines, columns=["Headline"])
df.to_csv("bbc_tech_headlines.csv", index=False)

# 4. Veriyi vektörleştirme
if cleaned_headlines:
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(cleaned_headlines)
    
    # Örnek etiketleme (etiketler rastgele oluşturuldu)
    y = [0 if i % 2 == 0 else 1 for i in range(len(cleaned_headlines))]
    
    # 5. Veriyi eğitim ve test setine ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 6. Farklı modelleri eğitme
    models = {
        "Naive Bayes": MultinomialNB(),
        "Lojistik Regresyon": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        print(f"{name} Model Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))
        
        # 7. Confusion Matrix Görselleştirme
        plt.figure(figsize=(5, 4))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Tahmin Edilen")
        plt.ylabel("Gerçek")
        plt.title(f"{name} Modeli İçin Confusion Matrix")
        plt.show()
    
    # 8. Sonuçları karşılaştırma
    plt.figure(figsize=(8, 5))
    plt.bar(results.keys(), results.values(), color=["blue", "green", "red"])
    plt.xlabel("Model")
    plt.ylabel("Doğruluk Oranı")
    plt.title("Farklı Makine Öğrenmesi Modellerinin Karşılaştırılması")
    plt.show()
    
    # 9. Kapsamlı Sonuç Analizi
    print("Kapsamlı Sonuç Analizi:")
    for name, accuracy in results.items():
        print(f"{name} Modeli Doğruluk Oranı: {accuracy:.4f}")
        
    # En iyi model belirleme
    best_model = max(results, key=results.get)
    print(f"En iyi performans gösteren model: {best_model} ({results[best_model]:.4f} doğruluk)")
    
    # Model performansları üzerine yorumlar
    def model_analysis():
        print("Naive Bayes modeli genellikle metin sınıflandırma problemlerinde iyi performans gösterir, ancak veri dağılımı dengesizse zayıf kalabilir.")
        print("Lojistik Regresyon, doğrusal olarak ayrılabilen verilerde güçlüdür ve genellikle dengeli sonuçlar verir.")
        print("Random Forest, daha karmaşık desenleri yakalamada iyidir ancak aşırı öğrenme riski taşır.")
        print("Genel olarak, model seçiminde doğruluk tek başına yeterli değildir; Precision, Recall ve F1-score gibi ek metrikler de dikkate alınmalıdır.")
    
    model_analysis()
    print("Veri kazıma tamamlandı, analiz yapıldı, modeller eğitildi, karşılaştırıldı ve kapsamlı bir sonuç analizi yapıldı!")
else:
    print("Hata: BBC'den haber başlıkları alınamadı. Lütfen URL'yi ve HTML etiketlerini kontrol edin.")
