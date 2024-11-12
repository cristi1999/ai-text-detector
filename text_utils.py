import string
import nltk
import spacy
import re
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag, word_tokenize
from textblob import TextBlob
from collections import Counter

nltk.download("punkt")
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))
nlp = spacy.load("en_core_web_sm")


def count_words(text):
    return len(text.split())


def count_chars(text):
    return len(text)


def count_punctuation(text):
    return sum(1 for char in text if char in string.punctuation)


def count_sentences(text):
    return len(sent_tokenize(text))


def avg_word_length(text):
    words = text.split()
    return sum(len(word) for word in words) / len(words)


def count_unique_words(text):
    return len(set(text.split()))


def count_stopwords(text):
    return sum(1 for word in text.split() if word.lower() in stop_words)


def avg_sentence_length(text):
    sentences = sent_tokenize(text)
    return (
        sum(len(sentence.split()) for sentence in sentences) / len(sentences)
        if sentences
        else 0
    )


def avg_word_length_no_punctuation(text):
    words = re.findall(r"\b\w+\b", text)
    return sum(len(word) for word in words) / len(words) if words else 0


def subjectivity_score(text):
    return TextBlob(text).sentiment.subjectivity


def sentiment_score(text):
    return TextBlob(text).sentiment.polarity


# words that appear only once
def hapax_legomena(text):
    words = text.split()
    frequency = Counter(words)
    return len([word for word, count in frequency.items() if count == 1])


def type_token_ratio(text):
    words = text.split()
    return len(set(words)) / len(words) if words else 0


def named_entity_count(text):
    doc = nlp(text)
    return len(doc.ents)


def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def generate_features(df, text_column):
    df["word_count"] = df[text_column].apply(count_words)
    df["char_count"] = df[text_column].apply(count_chars)
    df["punctuation_count"] = df[text_column].apply(count_punctuation)
    df["sentence_count"] = df[text_column].apply(count_sentences)
    df["avg_word_length"] = df[text_column].apply(avg_word_length)
    df["unique_word_count"] = df[text_column].apply(count_unique_words)
    df["stopword_count"] = df[text_column].apply(count_stopwords)
    df["avg_sentence_length"] = df[text_column].apply(avg_sentence_length)
    df["hapax_legomena"] = df[text_column].apply(hapax_legomena)
    df["type_token_ratio"] = df[text_column].apply(type_token_ratio)
    return df
