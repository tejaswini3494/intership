"""Level 3 - Task 2: NLP Sentiment Analysis."""

from collections import Counter
from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.tokenize import wordpunct_tokenize
from textblob import TextBlob
from wordcloud import WordCloud

BASE_DIR = Path(__file__).resolve().parent
RAW_FILE = BASE_DIR / "data" / "raw" / "Sentiment dataset.csv"
PROCESSED_FILE = BASE_DIR / "data" / "processed" / "sentiment_dataset.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"

STOP_WORDS = set(
    """
    i me my myself we our you your he him his she her it its they them their
    what which who this that these those am is are was were be been being have has had
    do does did a an the and but if or because as until while of at by for with about
    against between into through during before after above below to from up down in out
    on off over under again further then once here there all any both each few more most
    other some such no nor not only own same so than too very can will just dont should now
    """.split()
)

STEMMER = PorterStemmer()


def preprocess_text(text: str) -> tuple[list[str], list[str]]:
    """Tokenize, remove stopwords, and stem alphabetic tokens."""
    tokens = wordpunct_tokenize(str(text).lower())
    tokens = [
        word for word in tokens
        if re.fullmatch(r"[a-z]+", word) and len(word) > 1
    ]
    tokens = [word for word in tokens if word not in STOP_WORDS]
    stems = [STEMMER.stem(word) for word in tokens]
    return tokens, stems


def classify_sentiment(text: str) -> str:
    """Classify sentiment using TextBlob polarity."""
    polarity = TextBlob(str(text)).sentiment.polarity
    if polarity > 0.10:
        return "Positive"
    if polarity < -0.10:
        return "Negative"
    return "Neutral"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_FILE)
    df = df.drop_duplicates().reset_index(drop=True)

    text_col = "Text"
    df[text_col] = df[text_col].fillna("")

    df[["tokens", "stemmed_text"]] = pd.DataFrame(
        df[text_col].map(preprocess_text).tolist(),
        index=df.index,
    )

    df["polarity"] = df[text_col].map(
        lambda text: TextBlob(str(text)).sentiment.polarity
    )
    df["predicted_sentiment"] = df[text_col].map(classify_sentiment)

    counts = (
        df["predicted_sentiment"]
        .value_counts()
        .reindex(["Positive", "Negative", "Neutral"], fill_value=0)
    )

    print("Dataset shape:", df.shape)
    print("\nPredicted sentiment counts:\n", counts)

    plt.figure(figsize=(7, 5))
    counts.plot(kind="bar")
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Texts")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "sentiment_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    all_words = [word for row in df["tokens"] for word in row]
    top_words = pd.DataFrame(
        Counter(all_words).most_common(20),
        columns=["word", "frequency"],
    )
    top_words.to_csv(OUTPUT_DIR / "top_words.csv", index=False)

    if all_words:
        wordcloud = WordCloud(width=1000, height=500, background_color="white")
        wordcloud.generate(" ".join(all_words))
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title("Word Cloud - All Text")
        plt.savefig(PLOT_DIR / "word_cloud_all_text.png", dpi=300, bbox_inches="tight")
        plt.close()

    for sentiment in ["Positive", "Negative", "Neutral"]:
        words = [
            word
            for tokens, label in zip(df["tokens"], df["predicted_sentiment"])
            if label == sentiment
            for word in tokens
        ]
        if not words:
            continue

        wc = WordCloud(width=1000, height=500, background_color="white")
        wc.generate(" ".join(words))
        plt.figure(figsize=(12, 6))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud - {sentiment}")
        plt.savefig(
            PLOT_DIR / f"word_cloud_{sentiment.lower()}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    df.to_csv(PROCESSED_FILE, index=False)
    df.to_csv(OUTPUT_DIR / "sentiment_analysis_results.csv", index=False)

    print(f"\nProcessed data: {PROCESSED_FILE}")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
