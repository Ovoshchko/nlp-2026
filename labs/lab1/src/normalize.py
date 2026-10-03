import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer
import pymorphy3

nltk.download("stopwords", quiet=True)

stemmer_ru = SnowballStemmer('russian')
morph = pymorphy3.MorphAnalyzer()

STOP_WORDS = set(stopwords.words("russian"))
SPECIAL_TOKENS = {"<s>", "</s>", "[URL]", "[EMAIL]", "[NUM]", "[UNK]"}

def process_text(text: str) -> list[str]:
    tokens = re.findall(
        r"<s>|</s>|\[(?:URL|EMAIL|NUM|UNK)\]|[^\W\d_]+",
        text,
    )

    return [
        token if token in SPECIAL_TOKENS else token.lower()
        for token in tokens
        if token in SPECIAL_TOKENS or token.lower() not in STOP_WORDS
    ]


def prepare_tokens(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    df_prepared = df.copy()

    df_prepared[column_name] = df_prepared[column_name].apply(process_text)
    return df_prepared


def stem_dataframe(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    result = df.copy()

    result[column_name] = result[column_name].apply(
        lambda tokens: [
            word if word in SPECIAL_TOKENS else stemmer_ru.stem(word)
            for word in tokens
        ]
    )

    return result

def lemmatize_dataframe(
    df: pd.DataFrame,
    column_name: str,
) -> pd.DataFrame:
    result = df.copy()

    result[column_name] = result[column_name].apply(
        lambda tokens: [
            word if word in SPECIAL_TOKENS
            else morph.parse(word)[0].normal_form
            for word in tokens
        ]
    )

    return result