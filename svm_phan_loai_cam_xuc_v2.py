# -*- coding: utf-8 -*-


import os
import re
import pandas as pd
import numpy as np
from underthesea import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


if not os.path.exists("dataset.csv"):
    raise FileNotFoundError(
        "Khong tim thay file dataset.csv. Hay dat file dataset.csv "
        "cung thu muc voi file script nay truoc khi chay."
    )

df = pd.read_csv("dataset.csv", encoding="utf-8")
df = df.dropna(subset=["comment", "label"])  # loai bo dong thieu du lieu

VALID_LABELS = {"positive", "negative"}
if not set(df["label"]).issubset(VALID_LABELS):
    raise ValueError(
        "Cot 'label' trong dataset.csv chi duoc chua gia tri "
        "'positive' hoac 'negative'. Vui long kiem tra lai file du lieu."
    )

print("===== A. THONG TIN DU LIEU =====")
print("Tong so binh luan:", len(df))
print(df["label"].value_counts())
print(df.head(), "\n")


VIETNAMESE_STOPWORDS = {
    "va", "la", "cua", "thi", "nhung", "voi", "rat", "khi", "se",
    "da", "cho", "nay", "mot", "nhu", "neu", "the", "cac", "duoc",
}

def preprocess_text(text: str) -> str:
 
    """
    text = text.strip()                                    # B1: chuan hoa so bo
    text = text.lower()                                     # B2: chu thuong
    text = re.sub(r"[^a-zA-ZÀ-ỹà-ỹ0-9\s]", " ", text)        # B3: bo ky tu dac biet
    text = re.sub(r"\d+", " ", text)                         # B4: bo so
    text = re.sub(r"\s+", " ", text).strip()                 # B5: bo khoang trang thua

    tokenized = word_tokenize(text, format="text")           # B6: tach tu tieng Viet chuan
    tokens = tokenized.split()
    tokens = [t for t in tokens if t not in VIETNAMESE_STOPWORDS]  # B7: bo stopword

    return " ".join(tokens)

df["clean_comment"] = df["comment"].apply(preprocess_text)

# --- B2. Bieu dien van ban bang TF-IDF (theo muc 2.6 cua bao cao) ---
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df["clean_comment"])
y = df["label"]

# --- B3. Chia tap huan luyen / kiem tra ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- B4. Huan luyen mo hinh SVM (theo muc 2.3 cua bao cao) ---
model = SVC(kernel="linear", C=1.0)
model.fit(X_train, y_train)



# --- C1. Du doan tren tap kiem tra va danh gia mo hinh ---
y_pred = model.predict(X_test)

print("===== C. KET QUA DANH GIA MO HINH =====")
print("Accuracy :", round(accuracy_score(y_test, y_pred), 3))
print("Precision:", round(precision_score(y_test, y_pred, pos_label="positive"), 3))
print("Recall   :", round(recall_score(y_test, y_pred, pos_label="positive"), 3))
print("F1-score :", round(f1_score(y_test, y_pred, pos_label="positive"), 3))
print("\nBao cao chi tiet:\n", classification_report(y_test, y_pred))

# --- C2. Ve ma tran nham lan (confusion matrix) ---
cm = confusion_matrix(y_test, y_pred, labels=["positive", "negative"])
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["positive", "negative"],
            yticklabels=["positive", "negative"])
plt.xlabel("Du doan")
plt.ylabel("Thuc te")
plt.title("Ma tran nham lan - Mo hinh SVM")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("\nDa luu hinh ma tran nham lan: confusion_matrix.png")
plt.show()

# --- C3. Ham du doan cam xuc cho mot binh luan moi ---
def predict_sentiment(comment: str) -> str:
    cleaned = preprocess_text(comment)
    vector = tfidf.transform([cleaned])
    return model.predict(vector)[0]

print("\n===== C. DU DOAN BINH LUAN MOI =====")
new_comments = [
    "San pham rat dep va chat luong tot",
    "Giao hang cham, san pham khong nhu mong doi",
]
for cmt in new_comments:
    ket_qua = predict_sentiment(cmt)
    print(f"Binh luan: '{cmt}'  ->  Cam xuc du doan: {ket_qua}")
