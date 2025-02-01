# BBC News Veri Kazıma ve Makine Öğrenmesi Projesi

## Proje Hakkında  
Bu proje, BBC News web sitesinin teknoloji haberlerinden **veri kazıma (web scraping)** yöntemleri kullanılarak başlıkların toplandığı, bu verilerin temizlendiği ve üzerinde farklı **makine öğrenmesi algoritmalarının** uygulandığı bir çalışmadır. Proje süreci, veri kazıma, veri temizleme, model eğitimi ve sonuç değerlendirmesi gibi aşamalardan oluşmaktadır.

## Adımlar:  
1. **Veri Kazıma:** BBC News web sitesinin teknoloji haberlerinden başlıklar kazındı.  
2. **Veri Temizleme:** Kazınan veriler, analiz edilebilir bir formata dönüştürüldü.  
3. **Makine Öğrenmesi Algoritmaları:** Veri üzerinde 3 farklı makine öğrenmesi algoritması uygulandı.  
4. **Sonuçların Değerlendirilmesi:** Modellerin performansı ölçülüp karşılaştırıldı.

## Kullanılan Teknolojiler:
- **Python**
- **BeautifulSoup** (Veri Kazıma)
- **Pandas** (Veri Manipülasyonu)
- **Scikit-learn** (Makine Öğrenmesi)
- **Matplotlib**, **Seaborn** (Veri Görselleştirme)

## Proje Yapısı:
- `data/`: Veri kümesinin depolandığı klasör.  
- `scripts/`: Veri kazıma, veri temizleme ve modelleme adımlarının kodlarının bulunduğu klasör.  
- `results/`: Analiz sonuçları ve model değerlendirme raporları.  
- `README.md`: Projenin açıklamaları ve adımları hakkında bilgi veren dosya.

---

### Adımlar:

#### 1. Veri Kazıma:
BBC News web sitesinden haber başlıkları kazındı. Veriler, **BeautifulSoup** kütüphanesi ile HTML sayfalardan çıkarıldı.

```
URL = "https://www.bbc.com/news/technology"
headers = {"User-Agent": "Mozilla/5.0"}
response = requests.get(URL, headers=headers)
soup = BeautifulSoup(response.text, "html.parser")
headlines = [headline.text.strip() for headline in soup.find_all("h3")]
```

#### 2. Veri Temizleme
Kazınan veriler, aşağıdaki işlemlerle temizlendi:
- **Metinler küçük harfe çevrildi.**
- **Özel karakterler ve sayılar kaldırıldı.**
- **Gereksiz kelimeler (stopwords) temizlendi.**

```python
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

cleaned_headlines = [clean_text(headline) for headline in headlines]
df = pd.DataFrame(cleaned_headlines, columns=["Headline"])
df.to_csv("bbc_tech_headlines.csv", index=False)
```


#### 3. Makine Öğrenmesi Modelleri:

Veriler, TfidfVectorizer ile sayısal vektörlere dönüştürüldü ve üç farklı model eğitildi:

**Naive Bayes (MultinomialNB)**

**Lojistik Regresyon (LogisticRegression)**

**Random Forest (RandomForestClassifier)**

```
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(cleaned_headlines)

models = {
    "Naive Bayes": MultinomialNB(),
    "Lojistik Regresyon": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Model Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))
```

#### 4. Sonuçları Karşılaştırma ve Görselleştirme:

Model doğrulukları karşılaştırıldı ve Confusion Matrix görselleştirildi.

```
plt.figure(figsize=(8, 5))
plt.bar(results.keys(), results.values(), color=["blue", "green", "red"])
plt.xlabel("Model")
plt.ylabel("Doğruluk Oranı")
plt.title("Farklı Makine Öğrenmesi Modellerinin Karşılaştırılması")
plt.show()

```


```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# Model tahminlerini ve doğru etiketleri al
y_true = [0, 1, 0, 1]  # Gerçek etiketler
y_pred = [0, 1, 0, 1]  # Tahmin edilen etiketler

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

# Görselleştirme
plt.figure(figsize=(8, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
```



#### 5. Sonuçlar:

Projede elde edilen makine öğrenmesi modellerinin performans sonuçları model_performance.csv ve model_comparison.pdf dosyalarında sunulmuştur. 
En iyi model, en yüksek doğruluğa sahip olan Random Forest modelidir.


#### 6. Kullanım:


 **1.Depoyu klonlayın:**
   ``` 
   git clone https://github.com/yourusername/your-repository.git
   ```

 **2.Gereksinimleri yükleyin:**
    ```
    pip install -r requirements.txt
    ```
    
 **3.Veri kazıma işlemi için scrape_data.py dosyasını çalıştırın:**
    ```
    python scripts/scrape_data.py
    ```
    
 **4.Makine öğrenmesi modellerini çalıştırmak için model.py dosyasını çalıştırın:**
    ```
    python scripts/model.py
    ```

#### 7.Video Açıklama:

Proje hakkında detaylı açıklamalar ve sonuçların izahı için hazırladığım 3 dakikalık video açıklamasına aşağıdaki bağlantıdan erişebilirsiniz:

**[Video Açıklama Linki]**


## Lisans:
**Bu proje MIT Lisansı altında lisanslanmıştır.**



![Makine Öğrenmesi](https://github.com/user-attachments/assets/12448737-b2b1-4fdf-8d46-a244e27a382b)




![Makine Öğrenmesi 2](https://github.com/user-attachments/assets/6a8fc698-e1c9-437c-bd4d-831e3e38dbcd)



![Confusion Matrix Görseli](https://github.com/user-attachments/assets/7f18650c-4cc2-4ef6-b804-83d2755f07b2)







