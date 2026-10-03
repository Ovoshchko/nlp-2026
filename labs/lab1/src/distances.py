import pandas as pd
from collections import Counter, defaultdict

SPECIAL_TOKENS = {"<s>", "</s>", "[URL]", "[EMAIL]", "[NUM]", "[UNK]"}


def calculate_levenshtein(
    df: pd.DataFrame,
    column_name: str,
    top_n: int = 10,
) -> pd.DataFrame:
    frequencies = Counter(
        word
        for tokens in df[column_name]
        for word in tokens
        if word not in SPECIAL_TOKENS
    )

    words_by_length = defaultdict(list)
    for word in sorted(frequencies):
        words_by_length[len(word)].append(word)

    columns = [
        "Первое слово",
        "Второе слово",
        "Расстояние",
        "Частота первого",
        "Частота второго",
    ]
    results = []

    for length, words in sorted(words_by_length.items()):
        for i, first_word in enumerate(words):
            for other_length in range(length, length + 3):
                candidates = words_by_length.get(other_length, [])

                start = i + 1 if other_length == length else 0

                for j in range(start, len(candidates)):
                    second_word = candidates[j]
                    distance = calculate_levenshtein_for_pair(
                        first_word, second_word
                    )

                    if distance in (1, 2):
                        results.append({
                            "Первое слово": first_word,
                            "Второе слово": second_word,
                            "Расстояние": distance,
                            "Частота первого": frequencies[first_word],
                            "Частота второго": frequencies[second_word],
                        })

    results.sort(key=lambda pair: (
        pair["Расстояние"],
        -(pair["Частота первого"] + pair["Частота второго"]),
        pair["Первое слово"],
        pair["Второе слово"],
    ))

    return pd.DataFrame(results[:top_n], columns=columns)


def calculate_levenshtein_for_pair(first_word: str, second_word: str):
    if len(first_word) <= len(second_word):
        min_word, max_word = first_word, second_word
    else:
        min_word, max_word = second_word, first_word

    width = len(min_word)
    height = len(max_word)

    previous = list(range(width + 1))
    current = list(range(width + 1))
    
    for row in range(1, height + 1):
        current = [row] + [0] * width

        for column in range(1, width + 1):
            current[column] = min(previous[column] + 1, current[column - 1] + 1, previous[column - 1] + int(max_word[row - 1] != min_word[column - 1]))
        
        previous = current

    return previous[-1]

