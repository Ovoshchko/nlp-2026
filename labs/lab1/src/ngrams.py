import math
import re
import pandas as pd
from collections import Counter
import random

SPECIAL_TOKENS = {"<s>", "</s>", "[URL]", "[EMAIL]", "[NUM]", "[UNK]"}

def build_vocabulary(
    df: pd.DataFrame,
    column_name: str,
    min_count: int = 2,
) -> set[str]:
    counts = Counter(
        token
        for tokens in df[column_name]
        for token in tokens
    )

    vocabulary = {
        token
        for token, count in counts.items()
        if count >= min_count
    }

    return vocabulary | SPECIAL_TOKENS

def replace_unknowns(
    df: pd.DataFrame,
    column_name: str,
    vocabulary: set[str],
) -> pd.DataFrame:
    result = df.copy()

    result[column_name] = result[column_name].apply(
        lambda tokens: [
            token if token in vocabulary else "[UNK]"
            for token in tokens
        ]
    )

    return result

def make_three_gram(df: pd.DataFrame, column_name: str):
    trigram_counts = Counter()
    context_counts = Counter()

    for tokens in df[column_name]:
        padded_tokens = ["<s>"] + tokens

        for gram in get_grams(padded_tokens, n=3):
            trigram_counts[gram] += 1
            context_counts[gram[:2]] += 1

    return trigram_counts, context_counts

def get_grams(doc: list[str], n: int = 3) -> list[tuple[str, ...]]:
    return [
        tuple(doc[j:j + n])
        for j in range(len(doc) - n + 1)
    ]

def tokenize_for_model(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    result = df.copy()
    pattern = r"<s>|</s>|\[(?:URL|EMAIL|NUM|UNK)\]|[^\W\d_]+"
    result[column_name] = result[column_name].apply(
        lambda text: [
            token if token in SPECIAL_TOKENS else token.lower()
            for token in re.findall(pattern, text)
        ]
    )
    return result


def trigram_probability(
    context: tuple[str, str],
    word: str,
    trigram_counts: Counter,
    context_counts: Counter,
    vocabulary: set[str],
    alpha: float = 0.0,
) -> float:
    if word == "<s>":
        return 0.0
    context = tuple(w if w in vocabulary else "[UNK]" for w in context)
    word = word if word in vocabulary else "[UNK]"
    output_size = len(vocabulary) - int("<s>" in vocabulary)
    denominator = context_counts[context] + alpha * output_size
    if denominator == 0:
        return 0.0
    return (trigram_counts[context + (word,)] + alpha) / denominator


def calculate_perplexity(
    df: pd.DataFrame,
    column_name: str,
    trigram_counts: Counter,
    context_counts: Counter,
    vocabulary: set[str],
    alpha: float = 0.0,
) -> dict:
    log_likelihood = 0.0
    token_count = 0
    unseen_count = 0
    zero_count = 0

    for tokens in df[column_name]:
        mapped = [w if w in vocabulary else "[UNK]" for w in tokens]
        padded = ["<s>"] + mapped
        for i in range(2, len(padded)):
            context = tuple(padded[i - 2:i])
            word = padded[i]
            probability = trigram_probability(
                context, word, trigram_counts, context_counts, vocabulary, alpha
            )
            token_count += 1
            unseen_count += trigram_counts[context + (word,)] == 0

            if probability == 0:
                zero_count += 1
            else:
                log_likelihood += math.log(probability)

    if zero_count:
        log_likelihood = -math.inf
    entropy = -log_likelihood / token_count
    try:
        perplexity = math.exp(entropy)
    except OverflowError:
        perplexity = math.inf

    return {
        "Предсказанных токенов": token_count,
        "Невстречавшихся триграмм (вхождения)": unseen_count,
        "Нулевых вероятностей": zero_count,
        "Log-likelihood": log_likelihood,
        "Средний -log P": entropy,
        "Perplexity": perplexity,
    }

import random


def generate_text(
    trigram_counts,
    context_counts,
    vocabulary,
    max_length=50,
    alpha=1.0,
    rng=None,
):
    if rng is None:
        rng = random.Random()

    words = sorted(vocabulary - {"<s>"})
    context = ("<s>", "<s>")
    result = ["<s>"]

    for _ in range(max_length):
        weights = [
            trigram_counts[context + (word,)] + alpha
            for word in words
        ]

        if sum(weights) == 0:
            break

        word = rng.choices(words, weights=weights, k=1)[0]
        result.append(word)

        if word == "</s>":
            break

        context = (context[1], word)

    return " ".join(result)
