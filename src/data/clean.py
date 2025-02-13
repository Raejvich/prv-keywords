import re
from itertools import tee
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


def clean_text(full_text):
    # Remove special characters and keep hyphens
    cleaned_text = re.sub(r"[^a-zA-Z0-9\s-]", "", full_text)
    # lowercase
    cleaned_text = cleaned_text.lower()
    # tokenize with 2-gram
    tokenized_words = generate_ngrams(cleaned_text)
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    # Get the list of stopwords
    stop_words = set(stopwords.words("english"))
    # remove stopwords
    tokenized_words = [word for word in tokenized_words if word not in stop_words]
    # lemmatize
    cleaned_text = lemmatize_ngram(tokenized_words)
    return cleaned_text


def lemmatize_ngram(tokens):
    """
    Lemmatizes both individual words and n-grams from the list of tokens.

    Parameters
    ----------
    tokens : list of str
        List of tokens, which could include both single words and n-grams.

    Returns
    -------
    list
        List of lemmatized tokens, including both words and n-grams.
    """
    lemmatizer = WordNetLemmatizer()

    lemmatized_tokens = []

    for token in tokens:
        # If token is a word (single word, not an n-gram)
        if len(token.split()) == 1:
            lemmatized_tokens.append(lemmatizer.lemmatize(token))
        # If token is an n-gram (two or more words)
        else:
            # Lemmatize each word in the n-gram
            lemmatized_ngram = " ".join(
                lemmatizer.lemmatize(word) for word in token.split()
            )
            lemmatized_tokens.append(lemmatized_ngram)

    return lemmatized_tokens


def generate_ngrams(text, n=2):
    """
    Tokenizes a string of text into 1-grams and 2-grams.

    Parameters
    ----------
    text : str
        The input text to tokenize.
    n : int, optional
        The maximum n-gram size (default is 2).

    Returns
    -------
    list
        A list of 1-grams and 2-grams.
    """
    # Split the text into words
    words = text.split()

    # Generate 1-grams
    ngrams = words.copy()

    # Generate n-grams for n > 1 (up to 2 in this case)
    for size in range(2, n + 1):
        ngrams.extend(
            " ".join(words[i : i + size]) for i in range(len(words) - size + 1)
        )

    return ngrams


if __name__ == "__main__":
    text = "This is an runs example text! non-keywords are not considered?"
    cleaned_text = clean_text(text)
    print(cleaned_text)
