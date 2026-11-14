"""
Dashboard Analisis Sentimen - Kualitas Udara
Praktikum Neural Network (MLP) + Streamlit

Cara menjalankan (dari folder project, di terminal):
    pip install streamlit pandas numpy scikit-learn matplotlib seaborn wordcloud nltk
    streamlit run app_sentimen_nn.py

Data yang dibutuhkan:
    tweet_data.csv (kolom: created_at, username, full_text) di folder yang sama,
    atau upload CSV lewat sidebar.
"""

import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

try:
    from wordcloud import WordCloud
except ImportError:  # ditangani di tab WordCloud
    WordCloud = None


# ======================================================================
# KONSTANTA
# ======================================================================
DEFAULT_CSV = "tweet_data.csv"
MIN_TWEET = 300

URUTAN = ["positif", "netral", "negatif"]
WARNA = {"positif": "#2ca02c", "netral": "#7f7f7f", "negatif": "#d62728"}
COLORMAP = {"positif": "Greens", "netral": "Greys", "negatif": "Reds"}
EMOJI = {"positif": "😊", "netral": "😐", "negatif": "😠"}

KATA_POSITIF = {
    "bersih",
    "segar",
    "sehat",
    "baik",
    "bagus",
    "membaik",
    "pulih",
    "lega",
    "aman",
    "nyaman",
    "sejuk",
    "hijau",
    "asri",
    "lestari",
    "hujan",
    "berkurang",
    "menurun",
    "turun",
    "mengurangi",
    "menekan",
    "solusi",
    "upaya",
    "program",
    "peduli",
    "dukung",
    "mendukung",
    "apresiasi",
    "berhasil",
    "sukses",
    "harapan",
    "semangat",
    "pencegahan",
    "mencegah",
    "ramah",
    "lingkungan",
    "listrik",
    "transportasi",
    "umum",
    "sepeda",
    "tanam",
    "pohon",
    "terkendali",
    "cerah",
    "biru",
    "bebas",
}

KATA_NEGATIF = {
    "polusi",
    "tercemar",
    "pencemaran",
    "mencemari",
    "kotor",
    "buruk",
    "jelek",
    "parah",
    "bahaya",
    "berbahaya",
    "pekat",
    "kabut",
    "asap",
    "debu",
    "emisi",
    "sesak",
    "batuk",
    "ispa",
    "sakit",
    "gangguan",
    "penyakit",
    "menyengat",
    "bau",
    "pengap",
    "gerah",
    "merugikan",
    "rusak",
    "kerusakan",
    "darurat",
    "krisis",
    "menderita",
    "korban",
    "meninggal",
    "kecewa",
    "marah",
    "muak",
    "capek",
    "lelah",
    "macet",
    "gagal",
    "abuabu",
    "suram",
    "memburuk",
    "meningkat",
    "ancam",
    "mengancam",
}

# Kata umum yang disembunyikan dari WordCloud (tidak mengubah data/model)
STOPWORDS_WC = {
    "yang",
    "di",
    "dan",
    "ke",
    "dari",
    "ini",
    "itu",
    "untuk",
    "dengan",
    "pada",
    "adalah",
    "juga",
    "karena",
    "akan",
    "sudah",
    "saja",
    "tidak",
    "ada",
    "oleh",
    "dalam",
    "atau",
    "saat",
    "para",
    "agar",
    "bisa",
    "lebih",
    "masih",
    "telah",
    "yg",
    "nya",
    "dgn",
    "dg",
    "aja",
    "ga",
    "gak",
    "ya",
    "sih",
    "nih",
    "dong",
    "kok",
    "mereka",
    "kita",
    "kami",
    "saya",
    "terus",
    "hingga",
    "jadi",
    "lagi",
    "hari",
    "bukan",
    "belum",
    "jangan",
    "tapi",
    "bagi",
    "sampai",
    "kini",
    "udah",
    "bahkan",
    "amp",
    "guys",
    "eyes",
    "all",
    "on",
    "up",
    "dpr",
    "kualitas",
    "udara",
}

STOP_MANUAL = {
    "yang",
    "di",
    "dan",
    "ke",
    "dari",
    "ini",
    "itu",
    "untuk",
    "dengan",
    "pada",
    "adalah",
    "juga",
    "karena",
    "akan",
    "sudah",
    "saja",
    "ada",
    "oleh",
    "dalam",
    "atau",
    "saat",
    "para",
    "agar",
    "bisa",
    "lebih",
    "masih",
    "telah",
    "yg",
    "nya",
    "sebagai",
    "dapat",
    "bahwa",
    "sebuah",
    "kami",
    "kita",
    "mereka",
    "saya",
    "anda",
    "lagi",
    "pun",
}
# Kata negasi tidak dibuang karena penting untuk sentimen
NEGASI = {"tidak", "bukan", "jangan", "belum", "kurang"}

CONTOH_TEKS = [
    "Kualitas udara hari ini membaik setelah hujan, langit cerah dan udara terasa segar",
    "Polusi udara makin parah, banyak warga batuk dan terkena ISPA",
    "BMKG melaporkan indeks kualitas udara hari ini di Jakarta",
]


# ======================================================================
# PREPROCESSING & PELABELAN (sama dengan notebook)
# ======================================================================
def case_folding(text):
    if pd.isna(text):
        return ""
    return str(text).lower()


def cleaning(text):
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)  # URL
    text = re.sub(r"@\w+", " ", text)  # mention
    text = re.sub(r"#\w+", " ", text)  # hashtag + katanya
    text = re.sub(r"\d+", " ", text)  # angka
    text = re.sub(r"[^a-zA-Z\s]", " ", text)  # simbol, emoji, tanda baca
    text = re.sub(r"\s+", " ", text).strip()  # spasi berlebih
    return text


def preprocessing_pipeline(text):
    return cleaning(case_folding(text))


def skor_sentimen(tokens):
    pos = sum(1 for kata in tokens if kata in KATA_POSITIF)
    neg = sum(1 for kata in tokens if kata in KATA_NEGATIF)
    return pos, neg


def label_sentimen(tokens):
    pos, neg = skor_sentimen(tokens)
    if pos > neg:
        return "positif"
    if neg > pos:
        return "negatif"
    return "netral"  # seimbang atau tidak ada kata sentimen


# ======================================================================
# PENGOLAHAN DATA & MODEL (di-cache supaya dashboard tetap cepat)
# ======================================================================
@st.cache_data(show_spinner=False)
def proses_data(df_raw):
    """full_text -> teks_asli, case_folding, clean_text, sentimen."""
    df = df_raw.copy()
    df = df.rename(columns={"full_text": "teks_asli"})
    for kolom in ("created_at", "username"):
        if kolom not in df.columns:
            df[kolom] = ""
    df = df.dropna(subset=["teks_asli"]).reset_index(drop=True)

    df["case_folding"] = df["teks_asli"].apply(case_folding)
    df["clean_text"] = df["case_folding"].apply(cleaning)
    df["sentimen"] = df["clean_text"].apply(lambda t: label_sentimen(t.split()))
    return df[
        [
            "created_at",
            "username",
            "teks_asli",
            "case_folding",
            "clean_text",
            "sentimen",
        ]
    ]


@st.cache_resource(show_spinner=False)
def ambil_stopwords():
    try:
        import nltk

        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords

        daftar = set(stopwords.words("indonesian"))
    except Exception:
        daftar = set(STOP_MANUAL)
    daftar = {w for w in daftar - NEGASI if "-" not in w}
    return sorted(daftar)


def oversample(X, y, seed=42):
    """Oversampling kelas minoritas (hanya dipakai pada data latih)."""
    rng = np.random.RandomState(seed)
    X = pd.Series(X).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    maks = y.value_counts().max()
    idx = []
    for kelas in y.unique():
        ind = y[y == kelas].index.values
        idx.extend(ind)
        idx.extend(rng.choice(ind, maks - len(ind), replace=True))
    idx = rng.permutation(idx)
    return X.loc[idx], y.loc[idx]


@st.cache_resource(show_spinner=False)
def latih_model(df, n_hidden=100, pakai_oversampling=True):
    """Split 80:20 -> (oversampling) -> TF-IDF -> MLP (ReLU, Adam)."""
    df_model = df[df["clean_text"].astype(str).str.strip() != ""].copy()
    X = df_model["clean_text"].astype(str)
    y = df_model["sentimen"]

    # stratify hanya bisa dipakai bila tiap kelas punya minimal 2 data
    stratify = y if y.value_counts().min() >= 2 else None
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )
    except ValueError:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    n_latih_awal = len(X_tr)
    if pakai_oversampling:
        X_tr, y_tr = oversample(X_tr, y_tr)

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), sublinear_tf=True, stop_words=ambil_stopwords()
    )
    X_train = vectorizer.fit_transform(X_tr)
    X_test = vectorizer.transform(X_te)

    model = MLPClassifier(
        hidden_layer_sizes=(int(n_hidden),),
        activation="relu",
        solver="adam",
        alpha=0.001,
        max_iter=500,
        random_state=42,
    )
    model.fit(X_train, y_tr)
    y_pred = model.predict(X_test)

    return {
        "model": model,
        "vectorizer": vectorizer,
        "teks_uji": X_te,
        "y_test": y_te,
        "y_pred": y_pred,
        "teks_asli_uji": df_model.loc[X_te.index, "teks_asli"].values,
        "n_latih_awal": n_latih_awal,
        "n_latih": len(X_tr),
        "n_uji": len(X_te),
        "shape_train": X_train.shape,
        "shape_test": X_test.shape,
    }


@st.cache_data(show_spinner=False)
def gambar_wordcloud(teks, colormap):
    """Mengembalikan array gambar WordCloud (None jika teks kosong)."""
    if not teks.strip():
        return None
    wc = WordCloud(
        width=600,
        height=400,
        background_color="white",
        colormap=colormap,
        stopwords=STOPWORDS_WC,
        max_words=100,
    )
    return wc.generate(teks).to_array()


def hitung_metrik(y_test, y_pred):
    return pd.DataFrame(
        {
            "Metrik": [
                "Akurasi",
                "Presisi (weighted)",
                "Recall (weighted)",
                "F1-Score (weighted)",
            ],
            "Nilai": [
                accuracy_score(y_test, y_pred),
                precision_score(y_test, y_pred, average="weighted", zero_division=0),
                recall_score(y_test, y_pred, average="weighted", zero_division=0),
                f1_score(y_test, y_pred, average="weighted", zero_division=0),
            ],
        }
    ).round({"Nilai": 4})


def csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8-sig")


def isi_contoh(teks):
    st.session_state["input_teks"] = teks


# ======================================================================
# TAMPILAN DASHBOARD
# ======================================================================
def tab_dataset(df, df_raw):
    st.subheader("Dataset & Pemrosesan Teks")

    c1, c2, c3 = st.columns(3)
    c1.metric("Jumlah tweet", len(df))
    c2.metric(
        "Tweet dengan teks kosong setelah cleansing",
        int((df["clean_text"].str.strip() == "").sum()),
    )
    c3.metric("Kolom data mentah", len(df_raw.columns))
    if len(df) >= MIN_TWEET:
        st.caption(f"✅ Jumlah data memenuhi syarat minimal {MIN_TWEET} tweet.")
    else:
        st.warning(
            f"Jumlah data ({len(df)}) kurang dari syarat minimal {MIN_TWEET} tweet."
        )

    st.markdown("**Perbandingan teks asli dan teks bersih**")
    perbandingan = df[["teks_asli", "clean_text"]].rename(
        columns={"teks_asli": "Teks Asli", "clean_text": "Teks Bersih"}
    )
    st.dataframe(perbandingan.head(20), hide_index=True)

    st.markdown("**Dataset hasil pelabelan (rule-based)**")
    st.dataframe(
        df[["created_at", "username", "teks_asli", "clean_text", "sentimen"]].head(50),
        hide_index=True,
    )

    d1, d2 = st.columns(2)
    d1.download_button(
        "⬇️ Unduh CSV teks ternormalisasi",
        csv_bytes(
            df[["created_at", "username", "teks_asli", "case_folding", "clean_text"]]
        ),
        file_name="tweet_data_normalisasi.csv",
        mime="text/csv",
    )
    d2.download_button(
        "⬇️ Unduh CSV dataset berlabel",
        csv_bytes(df),
        file_name="dataset_kualitas_udara_berlabel.csv",
        mime="text/csv",
    )


def tab_distribusi(df):
    st.subheader("Distribusi Sentimen")
    distribusi = df["sentimen"].value_counts().reindex(URUTAN).fillna(0).astype(int)

    kolom = st.columns(3)
    for k, c in zip(URUTAN, kolom):
        persen = distribusi[k] / max(distribusi.sum(), 1) * 100
        c.metric(
            f"{EMOJI[k]} {k.capitalize()}",
            int(distribusi[k]),
            f"{persen:.1f}%",
            delta_color="off",
        )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(
        distribusi.index, distribusi.values, color=[WARNA[k] for k in distribusi.index]
    )
    axes[0].set_title("Distribusi Sentimen (Bar Chart)")
    axes[0].set_xlabel("Sentimen")
    axes[0].set_ylabel("Jumlah Tweet")
    for i, v in enumerate(distribusi.values):
        axes[0].text(i, v, str(v), ha="center", va="bottom")

    nonzero = distribusi[distribusi > 0]
    axes[1].pie(
        nonzero.values,
        labels=nonzero.index,
        autopct="%1.1f%%",
        colors=[WARNA[k] for k in nonzero.index],
        startangle=90,
    )
    axes[1].set_title("Distribusi Sentimen (Pie Chart)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def tab_wordcloud(df):
    st.subheader("WordCloud per Kelas Sentimen")
    if WordCloud is None:
        st.error(
            "Library `wordcloud` belum terpasang. Jalankan: `pip install wordcloud`, lalu muat ulang halaman."
        )
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, kelas in zip(axes, URUTAN):
        teks = " ".join(df.loc[df["sentimen"] == kelas, "clean_text"].astype(str))
        ax.set_title(f"WordCloud {kelas.capitalize()}", fontsize=14, color=WARNA[kelas])
        ax.axis("off")
        gambar = gambar_wordcloud(teks, COLORMAP[kelas])
        if gambar is None:
            ax.text(0.5, 0.5, "Tidak ada data", ha="center", va="center")
        else:
            ax.imshow(gambar, interpolation="bilinear")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        "Kata umum dan kata topik (kualitas, udara) disembunyikan agar kata khas tiap kelas lebih terlihat."
    )


def tab_evaluasi(hasil):
    st.subheader("Evaluasi Model Neural Network (MLP)")
    y_test, y_pred = hasil["y_test"], hasil["y_pred"]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Data latih (asli)", hasil["n_latih_awal"])
    k2.metric("Data latih (setelah oversampling)", hasil["n_latih"])
    k3.metric("Data uji", hasil["n_uji"])
    k4.metric("Fitur TF-IDF", hasil["shape_train"][1])

    tabel = hitung_metrik(y_test, y_pred)
    m = st.columns(4)
    for c, (_, baris) in zip(m, tabel.iterrows()):
        c.metric(baris["Metrik"], f"{baris['Nilai']:.4f}")

    kiri, kanan = st.columns(2)
    with kiri:
        st.markdown("**Tabel metrik performa**")
        st.dataframe(tabel, hide_index=True)
    with kanan:
        st.markdown("**Laporan per kelas**")
        laporan = pd.DataFrame(
            classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        ).T.round(3)
        st.dataframe(laporan)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].bar(
        tabel["Metrik"],
        tabel["Nilai"],
        color=["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"],
    )
    axes[0].set_ylim(0, 1.1)
    axes[0].set_title("Grafik Metrik Performa Model")
    axes[0].tick_params(axis="x", rotation=20)
    for i, v in enumerate(tabel["Nilai"]):
        axes[0].text(i, v + 0.02, f"{v:.2f}", ha="center")

    kelas_ada = [k for k in URUTAN if k in set(y_test) | set(y_pred)]
    cm = confusion_matrix(y_test, y_pred, labels=kelas_ada)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=axes[1],
        xticklabels=[f"Prediksi {k}" for k in kelas_ada],
        yticklabels=[f"Asli {k}" for k in kelas_ada],
    )
    axes[1].set_title("Confusion Matrix - Sentimen Kualitas Udara")
    axes[1].set_ylabel("Label Asli", labelpad=20)
    axes[1].set_xlabel("Prediksi Model")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("**Contoh hasil prediksi pada data uji**")
    contoh = pd.DataFrame(
        {
            "Teks Asli": hasil["teks_asli_uji"],
            "Sentimen Asli (y_test)": y_test.values,
            "Prediksi Model (y_pred)": y_pred,
        }
    )
    contoh["Benar?"] = np.where(
        contoh["Sentimen Asli (y_test)"] == contoh["Prediksi Model (y_pred)"],
        "✅",
        "❌",
    )
    st.dataframe(contoh.head(20), hide_index=True)

    with st.expander("Kurva loss pelatihan"):
        st.line_chart(pd.DataFrame({"loss": hasil["model"].loss_curve_}))

    st.caption(
        "Catatan: label dibuat otomatis dari kamus kata (rule-based), sehingga skor tinggi "
        "menunjukkan model berhasil meniru aturan pelabelan, bukan sentimen 'sebenarnya' dari manusia."
    )


def tab_prediksi(hasil):
    st.subheader("Prediksi Sentimen Teks Baru")
    model, vectorizer = hasil["model"], hasil["vectorizer"]

    st.write("Coba contoh cepat:")
    kolom = st.columns(len(CONTOH_TEKS))
    for i, (c, teks) in enumerate(zip(kolom, CONTOH_TEKS), start=1):
        c.button(f"Contoh {i}", key=f"contoh_{i}", on_click=isi_contoh, args=(teks,))

    teks = st.text_area(
        "Masukkan teks (tweet / kalimat) tentang kualitas udara:",
        key="input_teks",
        height=120,
        placeholder="Contoh: Polusi udara makin parah, banyak warga batuk dan terkena ISPA",
    )

    if not st.button("Prediksi Sentimen", type="primary"):
        return
    if not teks.strip():
        st.warning("Teks masih kosong, silakan isi dulu.")
        return

    bersih = preprocessing_pipeline(teks)
    if bersih == "":
        st.warning(
            "Tidak ada kata yang tersisa setelah pembersihan teks (mungkin hanya berisi angka/simbol/URL)."
        )
        return

    vektor = vectorizer.transform([bersih])
    label = str(model.predict(vektor)[0])
    proba = {
        str(k): float(v) for k, v in zip(model.classes_, model.predict_proba(vektor)[0])
    }

    st.markdown("**Teks setelah dibersihkan:**")
    st.code(bersih, language=None)

    hasil_teks = f"{EMOJI[label]}  Sentimen: {label.upper()}"
    if label == "positif":
        st.success(hasil_teks)
    elif label == "negatif":
        st.error(hasil_teks)
    else:
        st.info(hasil_teks)

    if vektor.nnz == 0:
        st.warning(
            "Tidak ada kata dalam teks ini yang dikenali model, jadi hasil prediksi kurang dapat diandalkan."
        )

    st.markdown("**Probabilitas tiap kelas:**")
    kelas_ada = [k for k in URUTAN if k in proba]
    kolom = st.columns(len(kelas_ada))
    for c, k in zip(kolom, kelas_ada):
        c.metric(f"{EMOJI[k]} {k.capitalize()}", f"{proba[k] * 100:.1f}%")
        c.progress(min(max(proba[k], 0.0), 1.0))

    pos, neg = skor_sentimen(bersih.split())
    st.caption(
        f"Pembanding rule-based: {label_sentimen(bersih.split())} "
        f"(kata positif: {pos}, kata negatif: {neg})."
    )


def main():
    st.set_page_config(
        page_title="Analisis Sentimen Kualitas Udara", page_icon="🌫️", layout="wide"
    )
    st.title("🌫️ Analisis Sentimen Kualitas Udara")
    st.caption(
        "Pipeline: Data Twitter → Cleansing → Pelabelan Rule-Based → Distribusi → WordCloud → "
        "Neural Network (MLP) → Evaluasi → Prediksi"
    )

    # ---------------- Sidebar ----------------
    st.sidebar.header("⚙️ Pengaturan")
    upload = st.sidebar.file_uploader(
        "Upload CSV (opsional)",
        type="csv",
        help="Kolom wajib: full_text. Kolom lain: created_at, username.",
    )
    n_hidden = st.sidebar.slider("Neuron lapisan tersembunyi", 10, 300, 100, step=10)
    pakai_os = st.sidebar.checkbox("Oversampling data latih", value=True)
    st.sidebar.markdown(
        "**Parameter tetap**\n"
        "- Split data: 80:20\n"
        "- Aktivasi: ReLU\n"
        "- Optimizer: Adam"
    )

    # ---------------- Muat data ----------------
    if upload is not None:
        df_raw = pd.read_csv(upload)
        sumber = f"file upload: {upload.name}"
    elif os.path.exists(DEFAULT_CSV):
        df_raw = pd.read_csv(DEFAULT_CSV)
        sumber = DEFAULT_CSV
    else:
        st.error(
            f"File `{DEFAULT_CSV}` tidak ditemukan di folder ini. Letakkan file di folder yang sama "
            "dengan `app_sentimen_nn.py`, atau upload CSV lewat sidebar."
        )
        st.stop()

    if "full_text" not in df_raw.columns:
        st.error(
            f"Kolom `full_text` tidak ada di data. Kolom yang ditemukan: {list(df_raw.columns)}"
        )
        st.stop()

    st.sidebar.success(f"Sumber data: {sumber}")

    df = proses_data(df_raw)
    if len(df) == 0:
        st.error("Data kosong setelah pembersihan.")
        st.stop()
    if df["sentimen"].nunique() < 2:
        st.error(
            "Data hanya menghasilkan satu kelas sentimen, model tidak bisa dilatih. "
            "Tambah data atau perluas kamus kata."
        )
        st.stop()

    with st.spinner("Melatih model Neural Network..."):
        hasil = latih_model(df, n_hidden, pakai_os)

    tabs = st.tabs(
        [
            "📄 Dataset",
            "📊 Distribusi",
            "☁️ WordCloud",
            "🧠 Evaluasi Model",
            "🔮 Prediksi Teks Baru",
        ]
    )
    with tabs[0]:
        tab_dataset(df, df_raw)
    with tabs[1]:
        tab_distribusi(df)
    with tabs[2]:
        tab_wordcloud(df)
    with tabs[3]:
        tab_evaluasi(hasil)
    with tabs[4]:
        tab_prediksi(hasil)


if __name__ == "__main__":
    main()