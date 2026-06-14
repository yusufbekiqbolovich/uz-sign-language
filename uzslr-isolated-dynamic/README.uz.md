# O'zbek Imo-ishora Tilini Tanish (UzSLR) — Izolyatsiyalangan Dinamik Imolar

> 🌐 Til / Language: **O'zbekcha** · [English](./README.md)

<p align="center">
  <img src="docs/gifs/inference_usage.gif" alt="Inference Demo" width="600">
</p>

Bu loyiha **50 ta izolyatsiyalangan, dinamik o'zbek imo-ishora tili (UzSL) imosini** veb-kamera videosidan real vaqtda taniydigan to'liq mashinaviy o'rganish (machine learning) tizimidir. Ushbu README butun jarayonni tushuntiradi — **ma'lumotlar qanday yig'ilgani, model qanday o'qitilgani va undan qanday foydalanish** — toki siz tizimni *tushunib*, boshqalarga ham *tushuntira* olishingiz uchun.

---

## 1. Umumiy tasvir (qisqacha)

Tizim odam imo qilayotgan qisqa veb-kamera videosini 50 ta so'zdan biriga aylantiradi. Bu beshta mantiqiy bosqichda amalga oshadi va har biri repozitoriydagi alohida papkaga to'g'ri keladi:

```
 Veb-kamera videosi                                          Bashorat qilingan imo
        │                                                              ▲
        ▼                                                              │
 ┌──────────────┐   ┌─────────────┐   ┌───────────────┐   ┌──────────┐ │
 │ MediaPipe    │ → │ 118 ta nuqta│ → │ Normalizatsiya│ → │ CNN +    │─┘
 │ Holistic     │   │ tanlanadi   │   │ + tezlik/     │   │ Transformer
 │ 543 ta nuqta │   │ (qo'l, yuz) │   │ tezlanish     │   │ → 50 sinf
 └──────────────┘   └─────────────┘   │ → 708 belgi   │   └──────────┘
                                       └───────────────┘
```

- Model imoni bir vaqtning o'zida **32 ta kadr (frame)** ko'rinishida ko'radi. Har bir kadr **708 ta sondan iborat belgi vektoriga** qisqartiriladi, va model **50 ta imo sinfi** bo'yicha ehtimollikni chiqaradi.
- Tanish modeli — ixcham (~1.75 million parametrli) **gibrid 1D-CNN + Transformer** bo'lib, Kaggle ASL musobaqasining 1-o'rin yechimidan (Sohn, 2023) moslashtirilgan va TensorFlow'dan **PyTorch'ga** ko'chirilgan.
- U ichki o'zbek ma'lumotlar to'plamida **~92% validatsiya** va **~87% test** aniqligiga erishadi.

### To'liq jarayon (pipeline)

| # | Bosqich | Papka | Vazifasi | Conda muhiti |
|---|---------|-------|----------|--------------|
| 1 | **Video yig'ish** | [`video-collector/`](./video-collector/) | Imo qiluvchilarni yozish, nuqtalarni ajratib olish | `video_collector_env` |
| 2 | **Ma'lumotni tayyorlash** | [`dataset-prep/`](./dataset-prep/) | Nuqtalarni qayta tashkillash, train/val/test ga bo'lish | `video_collector_env` |
| 3 | **Dastlabki ishlov (preprocessing)** | [`preprocessing/`](./preprocessing/) | Belgi tanlash, normalizatsiya, augmentatsiya | `uzslr-signs` |
| 4 | **Modelni o'qitish** | [`modeling/`](./modeling/) | CNN-Transformer'ni o'qitish va baholash | `uzslr-signs` |
| 5 | **Real vaqt inference** | [`inferencing/`](./inferencing/) | Desktop veb-kamera ilovasi (OpenCV oynasi) | `uzslr-signs` |
| 6 | **Veb-ilova** | [`web_app/`](./web_app/) | Brauzer ilovasi + ixtiyoriy LLM gap tuzish | `web-uzslr-signs` |

> 1–2 bosqichlar ma'lumotlar to'plamini quradi. 3–4 bosqichlar uni o'qitilgan modelga (`best_model.pth`) aylantiradi. 5–6 bosqichlar esa o'sha modeldan foydalanishning ikki xil usuli.
> Alohida eksperimental tarmoq, [`generative/`](./generative/), sun'iy imo-video yaratishni (DWPose + MusePose) ma'lumotni boyitish uchun o'rganadi — bu ixtiyoriy va asosiy jarayon qismi emas.

---

## 2. 50 ta imo

`assalomu_alaykum` · `bahor` · `birga` · `bo'sh` · `bosh_kiyim` · `boshlanishi` · `bozor` · `eshik` · `futbol` · `iltimos` · `internet` · `javob` · `jismoniy_tarbiya` · `karam` · `kartoshka` · `kichik` · `kitob` · `ko'prik` · `likopcha` · `maktab` · `mehmonxona` · `mehribon` · `metro` · `musiqa` · `o'simlik_yog'i` · `o'ynash` · `ochish` · `ot` · `ovqat_tayyorlash` · `oxiri` · `poezd` · `pomidor` · `qidirish` · `qish` · `qo'ziqorin` · `qor` · `qorong'i` · `quyon` · `restoran` · `sariyog'` · `shokolad` · `sovun` · `stakan` · `televizor` · `tosh` · `toza` · `turish` · `yomg'ir` · `yopish` · `yordam_berish`

Har bir imoning vizual namunasi (har biri ikki marta takrorlangan animatsion GIF) [`show-50-signs/`](./show-50-signs/) papkasida joylashgan — jonli modelni sinab ko'rganda imolarni o'zingiz bajarishingiz uchun.

---

## 3. O'rnatish va muhitlar

**Python:** `3.9.23` · **Anaconda:** `conda 24.11.3`

Loyiha ataylab **uchta alohida conda muhitidan** foydalanadi (vazifalarni ajratish prinsipi):

```bash
# A muhiti — ma'lumot yig'ish va tayyorlash (1–2 bosqich)
cd video-collector
conda env create -f environment-video-collector.yml
conda activate video_collector_env

# B muhiti — preprocessing, modeling, inferencing (3–5 bosqich)
conda env create -f environment-uzslr-signs.yml
conda activate uzslr-signs

# C muhiti — veb-ilova (6 bosqich)
cd web_app
conda env create -f environment-web-uzslr-signs.yml
conda activate web-uzslr-signs
```

> Ma'lumot to'plamini qayerga joylashtirish kerakligi haqida [`FOLDER_TREE.md`](./FOLDER_TREE.md) ga, aniq bog'liqliklar (dependency) uchun esa [`REPRODUCIBILITY.md`](./REPRODUCIBILITY.md) / [`CONDA_INFO.md`](./CONDA_INFO.md) ga qarang.

---

## 4. Ma'lumotlar qanday yig'ilgan — 1-bosqich ([`video-collector/`](./video-collector/))

**Maqsad:** real imo qiluvchilarni har bir imoni bajarayotganda yozib olish va ham videoni, ham har kadrdagi tana nuqtalarini (landmark) saqlash.

### Kim va qanday
- **Toshkentdagi 101-maktabda** o'tkazilgan. **10 ishtirokchi** (`signer01` … `signer10`) va imolar ma'nosini izohlab bergan **bir o'qituvchi**. Barcha ishtirokchilar to'liq xabardor qilingan va **yozma rozilik (consent) shakllarini imzolagan**.
- Maxsus **CLI yozib olish vositasi** ([`mod05_main.py`](./video-collector/mod05_main.py)) OpenCV kamera oqimini **MediaPipe Holistic** orqali o'tkazadi va u har kadrda **543 ta tana nuqtasini** aniqlaydi (468 yuz + 33 poza + 21 + 21 qo'l).

### Yozib olish jarayoni (3 bosqich)
| Bosqich | Harakat |
|---------|---------|
| **1. Imo qiluvchini tanlash** | `signerXX` id ni tanlash / yaratish |
| **2. Imoni tanlash** | raqamlangan ro'yxatdan so'z tanlash (yozilganlari yashil rangda) |
| **3. Yozib olish** | `s` ni bosish → hisob → aniq **32 ta kadr** yozib olinadi → qayta yozish uchun `s`, tugatish uchun `d` |

Har bir takror (repetition) — qat'iy **32 kadrli** klip. Aynan shu sababli model keyinchalik doimo 32 kadrli oynalar bilan ishlaydi.

### Nima saqlanadi
Har bir kadr **1662 ta `float32` qiymatdan** iborat NumPy fayl sifatida saqlanadi (`468×3` yuz + `33×4` poza + `2×21×3` qo'l), shuningdek xom video ham:

```
video-collector/Data_Numpy_Arrays_RSL_UzSL/
└── signer{XX}/
    └── {imo}/
        ├── landmarks/rep-{XX}/frame-00.npy … frame-31.npy   ← o'qitish uchun ishlatiladi
        └── videos/rep-{XX}/video.mp4                        ← faqat namuna uchun
```

> To'liq ma'lumot to'plami ~3 GB va u **`.gitignore`** ga kiritilgan (bu repozitoriyda yo'q). [`video-collector/dataset-checks/`](./video-collector/dataset-checks/) dagi skriptlar imo sonlari, takrorlar mosligi va har bir `.npy` aynan `(1662,)` shaklda ekanini tekshiradi.

**Ishga tushirish:** `cd video-collector && python mod05_main.py`

---

## 5. Ma'lumot qanday tayyorlangan — 2-bosqich ([`dataset-prep/`](./dataset-prep/))

Yig'uvchi tuzilmasi (imo qiluvchilar bo'yicha guruhlangan, videolar aralash) o'qitish uchun qulay emas, shuning uchun bu bosqich:

1. **Qayta tashkillaydi** — faqat nuqta `.npy` fayllarini `data/{imo}/rep-{XX}/` tekis papkasiga ko'chiradi ([`step01_reorganize_dataset.py`](./dataset-prep/step01_reorganize_dataset.py)).
2. **Bo'ladi** — takrorlarni **train (80%) / validation (10%) / test (10%)** ga ajratadi ([`step02_train_val_test_split.py`](./dataset-prep/step02_train_val_test_split.py)).

```
data/
├── train/{imo}/rep-{XX}/frame-XX.npy
├── validation/{imo}/rep-{XX}/frame-XX.npy
└── test/{imo}/rep-{XX}/frame-XX.npy
```

Ajratilgan **test to'plami 300 ta namunadan** iborat (50 imo × 6 takror). `dataset-checks/` dagi skriptlar bo'linishni tekshiradi.

> ⚠️ Bu skriptlar GB hajmdagi ma'lumotlarni ko'chiradi/ko'chiradi — avval [`dataset-prep/README.md`](./dataset-prep/README.md) ni diqqat bilan o'qing.

---

## 6. Belgilar qanday quriladi — 3-bosqich ([`preprocessing/`](./preprocessing/))

Bu tizimning yuragi: xom `(32, 1662)` nuqta kadrlarini model qabul qiladigan `(32, 708)` tenzorga aylantirish. **`Preprocess`** klassi orqali amalga oshiriladi ([`inferencing/inference02_preprocess.py`](./inferencing/inference02_preprocess.py) — joylashtirish uchun nusxasi).

### 6.1 Belgi tanlash — 543 → 118 ta nuqta
Poza nuqtalari foydali ma'lumot bermagani uchun faqat **118 ta nuqta** saqlab qolinadi:

| Guruh | Soni |
|-------|------|
| Lablar | 40 |
| Chap qo'l | 21 |
| O'ng qo'l | 21 |
| Burun | 4 |
| O'ng ko'z | 16 |
| Chap ko'z | 16 |
| **Jami** | **118** |

### 6.2 Nuqtalardan 708 ta belgigacha
Har bir 32 kadrli klip uchun:
1. `(1662,)` ni har kadr uchun `(543, 3)` ga ochish.
2. **118 ta tanlangan nuqtani** olish, faqat ularning **`x, y`** koordinatalarini saqlash → `(T, 118, 2)`.
3. Mos yozuvlar nuqtasiga (burun) nisbatan **markazlash va normalizatsiya**, vaqt bo'yicha standart og'ishga bo'lish.
4. **Tezlik** (1-tartib farq) va **tezlanish** (2-tartib farq) ni hisoblash.
5. **Pozitsiya + tezlik + tezlanish** ni birlashtirib, yassilash:

```
118 nuqta × 2 koordinata × 3 (pozitsiya, tezlik, tezlanish) = har kadrda 708 belgi
→ yakuniy namuna shakli: (32, 708)
```

Tezlik va tezlanish *statik* kadrga asoslangan modelga **harakatni** tushunish imkonini beradi — bu *dinamik* imolar uchun juda muhim.

### 6.3 Ma'lumotni boyitish — augmentatsiya (faqat o'qitishda)
Kichik ma'lumot to'plamining umumlashtirish qobiliyatini oshirish uchun belgilangan ehtimollik bilan tasodifiy qo'llaniladi:
- **Vaqtbo'yi qayta namunalash** (tezroq/sekinroq imolashni taqlid qilish)
- **Gorizontal aks ettirish** chap/o'ng nuqtalar almashinuvi bilan
- **Fazoviy affin** (masshtab, burilish, siljish, surilish)
- **Vaqt bo'yicha kesish** `MAX_LEN = 32` gacha
- **Fazoviy niqoblash** (to'rtburchak nuqta dropout)

> *Vaqt bo'yicha* niqoblash ataylab **ishlatilmaydi** — u harakat hosilalarida (derivative) uzilishlar yaratardi.

To'liq ma'lumot oqimi (PlantUML diagrammasi bilan) [`preprocessing/README.md`](./preprocessing/README.md) da hujjatlashtirilgan.

---

## 7. Model qanday o'qitilgan — 4-bosqich ([`modeling/`](./modeling/))

O'qitish va baholash [`modeling/notebooks/03_ak_model_dev_v1.ipynb`](./modeling/notebooks/03_ak_model_dev_v1.ipynb) notebook'ida joylashgan.

### 7.1 Arxitektura — gibrid 1D-CNN + Transformer
Kirish `(B, 32, 708)` → logitlar `(B, 50)`. [`inference03_model.py`](./inferencing/inference03_model.py) da aniqlangan (`SignLanguageModel`).

```
Kirish (32, 708)
   │
   ▼  Stem: Linear(708→dim) + BatchNorm
1-guruh: 3 × Conv1DBlock  +  1 × TransformerBlock
2-guruh: 3 × Conv1DBlock  +  1 × TransformerBlock
   │     (dim=384 katta model 3 va 4-guruhlarni qo'shadi)
   ▼
Top Linear (dim → 2·dim)
   ▼
32 kadr bo'yicha niqoblangan Global o'rtacha Pooling
   ▼
LateDropout (p=0.8) → Linear → 50 logit
```

Asosiy bloklar:
- **`Conv1DBlock`** — kengaytirish (Linear ×2) → SiLU → **kauzal depthwise Conv1D** (yadro 17) → BatchNorm → **ECA kanal e'tibori** → proyeksiya → residual. Kauzal konvolyutsiya qisqa muddatli vaqt naqshlarini ushlaydi.
- **`TransformerBlock`** — BatchNorm → **ko'p boshli o'z-o'ziga e'tibor** (4 bosh) → FFN (kengaytirish ×2) → residual. 32 kadr bo'ylab uzoq muddatli bog'lanishlarni ushlaydi.
- **ECA** = Efficient Channel Attention; **LateDropout** = faqat dastlabki bosqichdan keyin yoqiladigan kuchli dropout.

| Konfiguratsiya | `dim` | Guruhlar | Parametrlar |
|----------------|-------|----------|-------------|
| **Bazaviy** (yetkazilgan `best_model.pth`) | 192 | 2 | ~1.75 M |
| Katta | 384 | 4 | ~8–10 M |

### 7.2 O'qitish konfiguratsiyasi
| Sozlama | Qiymat |
|---------|--------|
| Loss (yo'qotish) | CrossEntropyLoss |
| Optimizator | AdamW (`lr=5e-4`, `weight_decay=0.1`) |
| Batch hajmi | 16 |
| Epoxalar | 300 gacha, **erta to'xtatish** (patience 15) |
| Augmentatsiya | flip, resample, affine, spatial mask |
| Apparat / vaqt | Apple MPS'da ~30 daqiqa |

### 7.3 Natijalar
- **~92% validatsiya aniqligi**, **~87% test aniqligi** (300 namunali test to'plami).
- Makro o'rtacha **F1 ≈ 0.87** (aniqlik 0.90 / qamrov 0.87).
- **Eng yaxshi sinflar** (F1 = 1.00): `maktab`, `musiqa`, `qidirish`, `qish`, `qorong'i`, `quyon`, `restoran`, `yomg'ir`, `yopish`, `o'ynash`.
- **Eng zaif sinflar:** `internet` (F1 0.29), `bozor` (0.50), va bir nechta vizual jihatdan o'xshash joy so'zlari (`metro`, `mehmonxona`, `poezd`) — to'liq sinf bo'yicha hisobot [`modeling/README.md`](./modeling/README.md) da.

**O'qitishni ishga tushirish:** `cd modeling/notebooks && jupyter notebook 03_ak_model_dev_v1.ipynb`
**Natijalar:** `best_model.pth` (eng yaxshi validatsiya aniqligi) va `checkpoint.pth` (davom ettirish nuqtasi).

---

## 8. Qanday foydalanish (A) — Real vaqt desktop ilovasi ([`inferencing/`](./inferencing/))

Veb-kameradan imolarni jonli taniydigan native OpenCV oynasi.

```bash
conda activate uzslr-signs        # yoki o'rnatish: pip install torch mediapipe opencv-contrib-python numpy
cd inferencing
python inference04_main.py
```

**Qanday ishlaydi:**
1. Kameraga **ikkala qo'lni** ko'rsating → holat `active` ga o'tadi va 32 kadrli bufer to'la boshlaydi (`buffer: N/32`).
2. Bufer 32 ga yetganda, bashorat qilingan imo + ishonch (confidence) yuqorida paydo bo'ladi, masalan `assalomu_alaykum (0.67)`.
3. Buferni qayta tiklash uchun qo'llaringizni 5 kadrdan ko'proq tushiring; chiqish uchun **`q`** ni bosing.

**Kamera eslatmasi:** kamera qurilmasi [`inference01_config.py`](./inferencing/inference01_config.py) dagi `VIDEO_DEVICE` orqali belgilanadi (hozir `0`). Agar veb-kamerangiz boshqa indeksda bo'lsa, shu yerda o'zgartiring. U MPS / CUDA / CPU ni avtomatik tanlaydi.

---

## 9. Qanday foydalanish (B) — Veb-ilova ([`web_app/`](./web_app/))

Brauzerga asoslangan versiya: **MediaPipe brauzerda ishlaydi**, nuqtalar WebSocket orqali FastAPI backendiga oqib boradi, va xuddi shu PyTorch modeli bashorat qiladi. Shuningdek, tanilgan imolarni grammatik to'g'ri o'zbek gapiga birlashtiradigan ixtiyoriy **LLM rejimi** ham mavjud (mahalliy Ollama modelidan foydalanib).

```bash
# Eng oson — Docker (faqat imo tanish)
docker pull 00015775/uzslr-web:1.0.0
docker run -p 7860:7860 00015775/uzslr-web:1.0.0      # → http://localhost:7860

# Mahalliy Python (faqat imo tanish)
conda activate web-uzslr-signs
uvicorn web_app.backend.main:app --port 8000          # → http://localhost:8000
```

LLM gap tuzish funksiyasi (`:2.0.0` image yoki `LLM_ENABLED=true`) va to'liq mahalliy/Docker ko'rsatmalari uchun [`web_app/README.md`](./web_app/README.md) ga qarang. Docker image'lar: <https://hub.docker.com/repository/docker/00015775/uzslr-web>.

---

## 10. Repozitoriya xaritasi

```
uzslr-isolated-dynamic/
├── video-collector/   # 1-bosqich — imolar + nuqtalarni yozish (CLI + MediaPipe)
├── dataset-prep/      # 2-bosqich — qayta tashkillash + train/val/test bo'lish
├── preprocessing/     # 3-bosqich — belgi muhandisligi notebook'lari
├── modeling/          # 4-bosqich — o'qitish notebook'i + best_model.pth, checkpoint.pth
├── inferencing/       # 5-bosqich — real vaqt desktop veb-kamera ilovasi
├── web_app/           # 6-bosqich — FastAPI + brauzer ilovasi (+ ixtiyoriy LLM)
├── generative/        # (ixtiyoriy) sun'iy imo-video yaratish (DWPose/MusePose)
├── show-50-signs/     # barcha 50 imo uchun namuna GIF'lar
├── docs/              # hujjatlarda ishlatilgan rasmlar va GIF'lar
└── *.md / *.yml       # yuqori darajadagi hujjatlar va conda muhit fayllari
```

---

## 11. Minnatdorchilik va manbalar

**Preprocessing va modeling jarayonlari** Kaggle "Isolated Sign Language Recognition" musobaqasining 1-o'rin yechimi — **Sohn, H. (2023)** (*Hoyso48* notebook'i) ning **mantiqiy moslashtirilishidir**. Asl nusxa ASL uchun **TensorFlow**'da yozilgan; bu yerda u **PyTorch**'da qayta amalga oshirilgan va boshqacha ma'lumot tuzilmasi bilan kam resursli **o'zbek** ma'lumot to'plamiga moslashtirilgan — bu **to'g'ridan-to'g'ri nusxa emas**.
Manba: [Sohn, H., 2023 – 1-o'rin yechimi (1D-CNN + Transformer)](https://www.kaggle.com/code/hoyso48/1st-place-solution-training).

**Toshkent 101-maktabiga** samimiy minnatdorchilik — imo tarjimalari, ma'nolari va kontekstini taqdim etgan 10 ishtirokchi va o'qituvchiga. Barcha ishtirokchilar **yozma ongli rozilik** berishgan; maxfiylik uchun `show-50-signs/` namunalarida odamlar ko'rsatilmagan (ular muallif tomonidan qayta bajarilgan).

<details>
<summary><b>Qo'shimcha manbalar</b></summary>

- Hoyso48 (2023). *1st place solution – training* [Kaggle notebook].
- Bergeron, M. (2024). *Insightful Datasets for ASL recognition*. Hackster.io.
- Cookiecutter Data Science — loyiha/nomlash shabloni.

</details>
