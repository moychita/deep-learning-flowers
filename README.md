# Klasifikasi Kucing vs Anjing: CNN vs Transfer Learning

Tugas individu Deep Learning. Membandingkan **CNN custom (dari nol)** dengan **Transfer Learning MobileNetV2** pada dataset gambar yang sama.

- **Nama:** (isi nama)
- **NIM:** (isi NIM)

## Dataset
[Cats and Dogs (filtered)](https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip): subset dari dataset Kaggle Dogs vs Cats.

| Split | Jumlah gambar |
|---|---|
| Train | 2.000 (1.000 kucing, 1.000 anjing) |
| Validation | 1.000 (500 kucing, 500 anjing) |

Gambar di-resize ke 160x160, batch size 32. Dataset diunduh otomatis oleh notebook.

## Arsitektur

**Model 1: CNN Custom**
`Augmentation -> Rescaling -> [Conv2D(32) + MaxPool] -> [Conv2D(64) + MaxPool] -> [Conv2D(128) + MaxPool] -> Flatten -> Dropout(0.5) -> Dense(128) -> Dense(1, sigmoid)`

**Model 2: Transfer Learning (MobileNetV2)**
`Augmentation -> Rescaling [-1,1] -> MobileNetV2 (ImageNet, include_top=False) -> GlobalAveragePooling -> Dropout(0.3) -> Dense(1, sigmoid)`
- Tahap 1: base model dibekukan (8 epoch, lr 1e-3)
- Tahap 2: fine-tuning 30 layer terakhir (5 epoch, lr 1e-5)

## Hasil

(Salin tabel dari bagian 8 notebook setelah dijalankan.)

| Model | Parameter | Train Acc | Val Acc | Gap | Waktu (s) |
|---|---|---|---|---|---|
| CNN Custom | ... | ... | ... | ... | ... |
| MobileNetV2 (TL) | ... | ... | ... | ... | ... |

Grafik: `images/training_curves.png`, `images/confusion_matrix.png`, `images/wrong_predictions.png`

## Analisis Singkat
(Isi 3-5 kalimat: model mana lebih baik, kenapa, kondisi overfitting, trade-off waktu dan ukuran.)

## Cara Menjalankan
**Google Colab (disarankan):** buka `notebook.ipynb`, pilih Runtime -> Change runtime type -> GPU (T4), lalu Run all.

**Lokal:**
```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

## Struktur Repository
```
.
├── README.md
├── notebook.ipynb
├── requirements.txt
└── images/
```
