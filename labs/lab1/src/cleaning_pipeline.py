import re
import pandas as pd


def replace_escaped_newlines(text: str) -> str:
    return text.replace(r"\r\n", " ").replace(r"\n", " ").replace(r"\r", " ")


def replace_urls(text: str) -> str:
    return re.sub(r"https?://[^\s]+|www\.[^\s]+", "[URL]", text)


def replace_emails(text: str) -> str:
    pattern = r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b"
    return re.sub(pattern, "[EMAIL]", text)


def replace_tabs(text: str) -> str:
    return re.sub(r"\t+", " ", text)


def replace_nonbreaking_spaces(text: str) -> str:
    return text.replace("\xa0", " ")


def replace_document_numbers(text: str) -> str:
    pattern = r"(?<!\w)(?:№|[Nn])\s*\d+(?:[-/][A-Za-zА-Яа-яЁё0-9]+)*"
    return re.sub(pattern, "[NUM]", text)


def replace_repeated_spaces(text: str) -> str:
    return re.sub(r" {2,}", " ", text)


def replace_numbers(text: str) -> str:
    return re.sub(r"\d+(?:[.,]\d+)*", "[NUM]", text)


def strip_text(text: str) -> str:
    return text.strip()

def add_boundary_tokens(text: str) -> str:
    return f"<s> {text} </s>"

def clean_text(text: str) -> str:
    steps = [
        replace_escaped_newlines,
        replace_urls,
        replace_emails,
        replace_tabs,
        replace_nonbreaking_spaces,
        replace_document_numbers,
        replace_numbers,
        replace_repeated_spaces,
        strip_text,
        add_boundary_tokens
    ]

    for step in steps:
        text = step(text)

    return text


def cleaning_pipeline(
    df: pd.DataFrame,
    column_name: str,
) -> pd.DataFrame:
    df_cleaned = df.copy()
    df_cleaned[column_name] = df_cleaned[column_name].apply(clean_text)
    return df_cleaned
