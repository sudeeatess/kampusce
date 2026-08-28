# 🎓 Kampüsce AI

### Offline University Knowledge & Verification Assistant

> **"Sorunu sor, üniversitenin kaynağından öğren."**

kampüsce AI, üniversite öğrencilerinin yönetmelik, staj yönergesi, sınav
kuralları, burs şartları ve akademik takvim gibi resmi belgeler içindeki
bilgiyi hızlıca bulmasını sağlayan, **tamamen offline çalışan** bir RAG
(Retrieval-Augmented Generation) asistanıdır. Hiçbir soru veya belge
internete/cloud servislere gönderilmez — tüm çıkarım **Microsoft
Foundry Local** üzerinden, kullanıcının kendi makinesinde gerçekleşir.

---

## 1. Problem

Öğrenciler; staj süresi, mazeret sınavı şartları, burs kriterleri gibi
sorularının cevabını onlarca sayfalık PDF yönetmelikler arasında arayarak
kaybeder. Genel amaçlı bir chatbot'a sormak ise "uydurma"
riski taşır — yanlış bir "staj süresi 20 gündür" cevabı öğrenciye ciddi
zarar verebilir.

## 2. Çözüm

kampüsce AI, **yalnızca kendi bilgi tabanındaki belgelerden cevap üretir.**
Her cevap üç durumdan biriyle etiketlenir:

| Durum | Anlamı |
|---|---|
| 🟢 **DOĞRULANDI** | Kaynaklarda soruya net ve açık cevap var |
| 🟡 **KISMEN DOĞRULANDI** | İlgili bilgi var ama soruya kesin cevap vermeye yetmiyor |
| 🔴 **BİLGİ BULUNAMADI** | Bilgi tabanında yeterli bilgi yok — sistem tahmin etmez |

Her cevabın yanında **kaynak belge, madde ve sayfa numarası** ile bir
**benzerlik yüzdesi** gösterilir.

## 3. Özellikler

- PDF / TXT / Markdown belge yükleme, madde/sayfa metadata'sı korunarak chunking
- Microsoft Foundry Local üzerinden gerçek local embedding + LLM çıkarımı
  (OpenAI uyumlu REST arayüzü, cloud API'siz)
- SQLite tabanlı bilgi tabanı + saf Python/NumPy cosine similarity retrieval
- Top-K ve benzerlik eşiği (relevance threshold) ile ilgisiz belgelerin elenmesi
- Hallucination'ı engelleyen katı RAG prompt mimarisi
- Modern,özgün arayüz 
- Karanlık/aydınlık tema, arama geçmişi, bilgi tabanı istatistikleri, ayarlar
- Güvenli dosya yükleme (uzantı kontrolü, boyut limiti, path-traversal koruması)
- Boş dosya / bozuk PDF / servis kapalı gibi durumlarda kullanıcı dostu hata mesajları

## 4. Teknolojiler

- **Backend:** Python 3.11+, FastAPI, Uvicorn, SQLite (`sqlite3`, ek bağımlılık yok)
- **AI çıkarımı:** Microsoft Foundry Local (`foundry-local-sdk`, OpenAI Python istemcisiyle local REST çağrısı)
- **Belge işleme:** `pypdf`
- **Embedding/matematik:** NumPy (cosine similarity)
- **Frontend:** Tek dosyaya gömülü HTML/CSS/Vanilla JS (framework yok, derleme adımı yok)

## 5. Mimari

```
kampüsceAI/
├── app.py                 ← TEK DOSYA: backend + gömülü frontend (bu proje burada)
├── requirements.txt
├── .env.example
├── documents/              ← yüklenen belgelerin fiziksel olarak saklandığı klasör
├── data/
│   └── kampüsce.db        ← SQLite veritabanı (ilk çalıştırmada otomatik oluşur)
├── ornek_belgeler/          ← demo için hazır örnek yönetmelik metinleri
└── README.md
```

`app.py` içindeki bölümler (yukarıdan aşağıya):

1. Config (.env okuma)
2. SQLite şeması ve bağlantı yönetimi (`documents`, `chunks`, `queries`, `answers`)
3. Belge okuma (PDF/TXT/MD)
4. Chunking (paragraf + "Madde N" sınırlarına duyarlı)
5. `FoundryClient` — Foundry Local ile iletişim (embedding + chat)
6. Ingestion pipeline (okuma → chunking → embedding → SQLite)
7. Retrieval (cosine similarity + top-K + threshold)
8. Prompt engineering + RAG pipeline + JSON ayrıştırma
9. FastAPI route'ları (`/api/...`)
10. Gömülü frontend (`FRONTEND_HTML` sabiti)

### RAG Pipeline

```
Kullanıcı Sorusu
      ↓
Query Embedding (Foundry Local)
      ↓
SQLite'taki tüm chunk embedding'leri
      ↓
Cosine Similarity (NumPy)
      ↓
Top-K + Similarity Threshold filtresi
      ↓
Context oluşturma (SOURCE 1 / SOURCE 2 ... formatı)
      ↓
Foundry Local LLM (katı system prompt: "sadece context'i kullan")
      ↓
JSON cevap: answer + verification_status + sources
```


## 6. Model Ayarları

`.env` dosyasındaki ilgili değişkenler:

```env
FOUNDRY_LLM_MODEL=            # boş bırakılırsa otomatik seçilir
FOUNDRY_EMBEDDING_MODEL=qwen3-embedding-0.6b
TOP_K=3
SIMILARITY_THRESHOLD=0.60
TEMPERATURE=0.2
```

`TOP_K`, `SIMILARITY_THRESHOLD` ve `TEMPERATURE` uygulama açıkken
**Ayarlar** sayfasından da değiştirilebilir (anlık olarak, `.env` dosyasını
bozmadan). Ayarları değiştirdikten sonra mevcut belgeleri yeni parametrelerle
yeniden işlemek isterseniz "Yeniden İndeksle" butonunu kullanın.

## 7. Veritabanı

SQLite, `data/kampüsce.db` dosyasında saklanır — ek bir veritabanı
sunucusu kurmanıza gerek yoktur. Şema:

- **documents** — yüklenen her belgenin meta verisi ve işlem durumu
- **chunks** — belge parçaları + embedding vektörleri (JSON olarak)
- **queries** — sorulan her soru
- **answers** — her sorunun cevabı, doğrulama durumu, güven skoru ve kaynakları

## 8. Kullanım

1. **Belgeler** sayfasından `ornek_belgeler/` klasöründeki örnek yönetmelikleri
   (veya kendi PDF/TXT/MD dosyalarınızı) yükleyin.
2. **Yeni Sorgu** sayfasında örnek sorulardan birine tıklayın veya kendi
   sorunuzu yazın.
3. Cevap; doğrulama durumu , güven yüzdesi ve tıklanabilir
   kaynak kartlarıyla birlikte gelir. "Teknik Detaylar" panelinden hangi
   modelin ve kaç chunk'ın kullanıldığını görebilirsiniz.

## 9. Senaryo

`ornek_belgeler/staj_yonergesi.txt` yüklendikten sonra:

| Soru | Beklenen Durum |
|---|---|
| "Staj süresi kaç iş günü?" | 🟢 DOĞRULANDI — Madde 8 |
| "3 dersten kalan öğrenci staj yapabilir mi?" | 🟡 KISMEN DOĞRULANDI — Madde 14 ilgili ama net değil |
| "Üniversitenin kantininde bugün ne yemek var?" | 🔴 BİLGİ BULUNAMADI |

Bu üç senaryo, sistemin temel değer önerisini gösterir: **cevap
verebildiğinde cevap verir, veremediğinde uydurmaz.**

## 10. Testler

Otomatik testler için `pytest` kullanılabilir (Foundry Local'ın çalışır
durumda olması gerekir — embedding/LLM çağrıları mock'lanmamıştır, bilinçli
bir tercihtir: proje "gerçek local inference" üzerine kuruludur). Manuel
test için:

- Boş soru → validation mesajı döner (`is_validation_error: true`)
- "Merhaba" gibi genel bir mesaj → gereksiz retrieval yapılmadan kibar bir
  karşılama döner
- Bilgi tabanı boşken soru sorma → `not_found` + "önce belge yükleyin" uyarısı
- Yanlış uzantılı dosya, path-traversal'lı dosya adı, boyut limiti aşan dosya
  → hepsi güvenli şekilde reddedilir.

## 11. Proje Sınırlılıkları

- Taranmış (image-based) PDF'lerde OCR desteği yoktur — metin katmanı
  olmayan PDF'ler okunamaz olarak işaretlenir.
- "Madde N" tespiti regex tabanlıdır; madde numaralandırması farklı bir
  formatta olan belgelerde (örn. "Article 5", "§5") madde bilgisi `None`
  dönebilir — chunk yine de içerik olarak doğru şekilde saklanır, sadece
  kaynak gösteriminde madde numarası eksik olur.
- Embedding'ler küçük/orta ölçekli bilgi tabanları için SQLite + NumPy
  cosine similarity ile hesaplanır; on binlerce chunk'a ölçeklenmesi
  gerekirse ayrı bir vector database (örn. FAISS, sqlite-vec) önerilir.
- Ayarlar sayfasındaki değişiklikler bellek içi (runtime) çalışır; sunucu
  yeniden başlatıldığında `.env` dosyasındaki değerlere döner.


**Öncelik sırası:** çalışması → doğru RAG mimarisi → kaynak gösterme →
hallucination kontrolü → profesyonel UI → test → dokümantasyon → ekstra özellikler.
