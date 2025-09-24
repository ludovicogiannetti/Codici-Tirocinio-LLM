import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.parse import urlparse
import re
from datasets import load_from_disk
from wordcloud import WordCloud
import spacy
from collections import Counter

def main():
    # Caricamento e conversione del dataset
    dataset = load_from_disk("laion_subset_10000")
    df = dataset.to_pandas()

    # Analisi iniziale
    print("Shape:", df.shape)
    print("\nColonne disponibili:", df.columns.tolist())
    print("\nValori mancanti per colonna:")
    print(df.isnull().sum())

    dataset_overview(df)

    # Analisi degli status
    analyze_status(df)

    # Pulizia e preprocessing
    df = clean_and_preprocess(df)

    # Sample dopo il preprocessing
    print("\nSample dopo preprocessing:")

    # Analisi dettagliate
    analyze_domains(df)
    analyze_captions(df)
    analyze_caption_text(df, lang="en")
    analyze_similarity(df)

    plt.show()

def dataset_overview(df):
    print("\n=== Panoramica Dataset ===")

    if 'caption' in df.columns:

        # Lunghezze di tutte le caption
        caption_lengths = df['caption'].dropna().apply(len)

        min_len = caption_lengths.min()
        max_len = caption_lengths.max()
        avg_len = caption_lengths.mean()

        # Numero totale di parole
        total_words = df['caption'].dropna().str.split().str.len().sum()        
        print(f"Numero totale di parole nelle caption: {total_words}")
        print(f"Lunghezza media caption: {avg_len:.2f} caratteri")
        print(f"Lunghezza minima caption: {min_len}")
        print(f"Lunghezza massima caption: {max_len}")


    # Immagini (dimensioni)
    if all(c in df.columns for c in ['width', 'height']):
        areas = df['width'] * df['height']
        avg_area = areas.mean()
        min_area = areas.min()
        max_area = areas.max()
        print(f"Dimensione media immagine (px): {avg_area:.0f}")
        print(f"Dimensione minima immagine (px): {min_area}")
        print(f"Dimensione massima immagine (px): {max_area}")

    # Se ci sono anche le dimensioni originali
    if all(c in df.columns for c in ['original_width', 'original_height']):
        orig_areas = df['original_width'] * df['original_height']
        print(f"Dimensione media immagine originale (px): {orig_areas.mean():.0f}")
        print(f"Dimensione minima immagine originale (px): {orig_areas.min()}")
        print(f"Dimensione massima immagine originale (px): {orig_areas.max()}")

def analyze_status(df):
    print("\nAnalisi degli status:")
    if 'status' in df.columns:
        status_counts = df['status'].value_counts(dropna=False)
        print("\nDistribuzione degli status:")
        print(status_counts)

        non_success = df[df['status'] != 'success']
        if len(non_success) > 0:
            print(f"\nTrovati {len(non_success)} record con status diverso da 'success':")

            if 'error_message' in non_success.columns:
                error_analysis = non_success['error_message'].value_counts().head(10)
                print("\nTop 10 messaggi di errore:")
                print(error_analysis)

            print("\nEsempi di record non-success:")
            print(non_success[['status', 'error_message', 'url']].head(3))

        else:
            print("\nTutti i record hanno status 'success'")
    else:
        print("\nLa colonna 'status' non è presente nel dataset")

def analyze_similarity(df):
    if "similarity" not in df.columns:
        print("Colonna similarity non trovata")
        return

    sim = df["similarity"].dropna()

    desc = sim.describe(percentiles=[.05, .25, .5, .75, .95])
    print("\nStatistiche similarity:\n", desc)
    
    plt.figure(figsize=(10,5))
    sns.histplot(sim, bins=50, kde=True)
    plt.title("Distribuzione similarity")
    plt.xlabel("Similarity")
    plt.ylabel("Frequenza")

def analyze_caption_text(df, lang="en"):

    if 'caption' not in df.columns or df['caption'].dropna().empty:
        print("\nNessuna caption disponibile.")
        return

    print("\nCaricamento modello NLP spaCy...")
    if lang == "en":
        nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    elif lang == "it":
        nlp = spacy.load("it_core_news_sm", disable=["ner", "parser"])
    else:
        raise ValueError("Lingua non supportata. Usa 'en' o 'it'.")

    texts = df['caption'].dropna().astype(str).sample(min(5000, len(df))).tolist()
    docs = nlp.pipe(texts, batch_size=1000, n_process=2)

    lemmas = []
    for doc in docs:
        for token in doc:
            lemma = token.lemma_.lower().strip()

            if (
                not token.is_stop 
                and not token.is_punct
                and lemma.isalpha()        
                and len(lemma) > 2         
            ):
                lemmas.append(lemma)

    counter = Counter(lemmas)
    top_lemmas = counter.most_common(20)
    
    print("\nTop 20 lemmi più frequenti:")

    for lemma, freq in top_lemmas:
        print(f"{lemma}: {freq}")

    plt.figure(figsize=(12,6))
    sns.barplot(x=[x[1] for x in top_lemmas], y=[x[0] for x in top_lemmas], palette="magma")
    plt.title("Top 20 lemmi più frequenti nelle caption")
    plt.xlabel("Frequenza")
    plt.ylabel("Lemma")

def clean_and_preprocess(df):
    if 'status' in df.columns:
        df = df[df['status'] == 'success'].copy()

    def safe_parse(url):
        try:
            return urlparse(str(url)).netloc if pd.notnull(url) else None
        except:
            return None

    df = df.assign(
        domain=df['url'].apply(safe_parse),
        extension=df['url'].str.extract(r'\.([a-zA-Z0-9]+)(?=[?#]|$)', flags=re.IGNORECASE),
        has_watermark=df['pwatermark'] > 0.5,
        is_unsafe=df['punsafe'] > 0.5
    )

    num_cols = ['similarity', 'width', 'height', 'original_width', 'original_height']
    for col in num_cols:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    if 'caption' in df.columns:
        df['caption'] = df['caption'].fillna('')
        df['caption_length'] = df['caption'].str.len()

    if all(col in df.columns for col in ['width', 'height', 'original_width', 'original_height']):
        df['dimension_diff'] = (df['original_width']*df['original_height']) - (df['width']*df['height'])

    if all(col in df.columns for col in ['width', 'height']):
        df['aspect_ratio'] = df['width'] / df['height'].replace(0, np.nan)

    return df

def analyze_domains(df):
    if 'domain' in df.columns:
        top_domains = df['domain'].value_counts().head(10)
        print("\nTop 10 domini:")
        print(top_domains)

        plt.figure(figsize=(10,5))
        sns.barplot(x=top_domains.values, y=top_domains.index, palette="viridis")
        plt.title("Top 10 domini di origine")
        plt.xlabel("Conteggio")
        plt.ylabel("Dominio")

def analyze_captions(df):
    if 'caption_length' in df.columns:
        plt.figure(figsize=(10,5))
        sns.histplot(df['caption_length'], bins=30, kde=True)
        plt.title("Distribuzione lunghezza caption")

    if 'caption' in df.columns and len(df['caption'].dropna()) > 0:
        text = " ".join(df['caption'].dropna().astype(str).sample(min(2000,len(df))))
        wc = WordCloud(width=800, height=400, background_color="white").generate(text)
        plt.figure(figsize=(12,6))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title("WordCloud caption")

if __name__ == "__main__":
    main()
