import pandas as pd
import re
import nltk
from collections import Counter
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("russian"))

SPECIAL_TOKENS = {"<s>", "</s>", "[URL]", "[EMAIL]", "[NUM]", "[UNK]"}

def count_words_by_char(df: pd.DataFrame, column_name: str):
    return pd.DataFrame(df[column_name].str.len())

def count_words_with_space_separator(df: pd.DataFrame, column_name: str):
    return pd.DataFrame(df[column_name].str.split().str.len())

def top_stop_words(texts, top_n=30):
    counts = Counter()
    total_words = 0

    for text in texts:
        words = re.findall(r"[а-яё]+", text.lower())
        total_words += len(words)
        counts.update(word for word in words if word in STOP_WORDS)

    if total_words == 0:
        return []

    return [
        (word, count / total_words)
        for word, count in counts.most_common(top_n)
    ]

import pandas as pd


def count_noise(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    patterns = {
        "Юзернеймы": r"(?<![\w@])@[A-Za-zА-Яа-яЁё0-9_]+",
        "Хештеги": r"(?<!\w)#[A-Za-zА-Яа-яЁё0-9_]+",
        "HTML-теги": r"<(?!/?s>)/?[A-Za-z][^>]*>",
        "URL": r"https?://[^\s]+|www\.[^\s]+",
        "Табуляции": r"\t+",
        "Неразрывные пробелы": r"\xa0",
        "Повторные пробелы": r" {2,}",
        "Почта (email)": r"[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b",
    }

    results = []

    for name, pattern in patterns.items():
        counts = df[column_name].str.count(pattern)

        results.append({
            "Тип": name,
            "Совпадений": counts.sum(),
            "Документов": (counts > 0).sum(),
            "Доля документов, %": (counts > 0).mean() * 100,
        })

    return pd.DataFrame(results)

def compare_vocabularies(
    before: pd.DataFrame,
    after: pd.DataFrame,
    column_name: str,
) -> pd.DataFrame:
    results = []

    for name, df in [("До", before), ("После", after)]:
        vocabulary = {
            token
            for tokens in df[column_name]
            for token in tokens
            if token not in SPECIAL_TOKENS
        }

        results.append({
            "Вариант": name,
            "Размер словаря": len(vocabulary),
        })

    return pd.DataFrame(results)