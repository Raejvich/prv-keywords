import gensim
from gensim import corpora, models
from gensim.models import CoherenceModel


# Define the function to find optimal number of topics and return topics with words
def find_optimal_topics_from_text(text, min_topics=2, max_topics=10):
    """
    Function to find the optimal number of topics using coherence score
    for given tokenized text (single words and bigrams in one list).

    Parameters:
    - text: List of tokenized words and bigrams (single list).
    - min_topics: Minimum number of topics to evaluate (default is 2).
    - max_topics: Maximum number of topics to evaluate (default is 10).

    Returns:
    - optimal_num_topics: The number of topics that produces the highest coherence score.
    - lda_model: Trained LDA model with the optimal number of topics.
    - topics_with_words: List of topics with each word and its weight.
    """
    # Step 1: Create a dictionary and corpus from the text
    dictionary = corpora.Dictionary([text])  # Pass a 2D array with a single document
    corpus = [dictionary.doc2bow(text)]  # Corpus is the bag-of-words representation

    # Step 2: Find optimal number of topics using coherence score
    coherence_scores = []
    topic_range = range(min_topics, max_topics + 1)

    for num_topics in topic_range:
        lda_model = models.LdaModel(
            corpus, num_topics=num_topics, id2word=dictionary, passes=10
        )
        coherence_model = CoherenceModel(
            model=lda_model, texts=[text], dictionary=dictionary, coherence="c_v"
        )
        coherence_scores.append(coherence_model.get_coherence())

    # Select the number of topics with the highest coherence score
    optimal_topics = topic_range[coherence_scores.index(max(coherence_scores))]

    # Step 3: Train the final LDA model with the optimal number of topics
    lda_model = models.LdaModel(
        corpus, num_topics=optimal_topics, id2word=dictionary, passes=10
    )

    # Step 4: Get the words for each topic with their weights
    topics_with_words = []
    for topic_id, topic_words in lda_model.print_topics(-1):
        words = [
            (word.split("*")[1].strip().strip('"'), float(word.split("*")[0]))
            for word in topic_words.split(" + ")
        ]
        topics_with_words.append((topic_id, words))

    return optimal_topics, lda_model, topics_with_words


# Example usage (if running as main script)
if __name__ == "__main__":
    # Sample tokenized text (single list with words and bigrams)
    text = [
        "run",
        "example",
        "text",
        "non-keywords",
        "considered",
        "this is",
        "is an",
        "an run",
        "run example",
    ]

    optimal_num_topics, lda_model, topics_with_words = find_optimal_topics_from_text(
        text
    )

    # Print the results
    print(f"Optimal number of topics: {optimal_num_topics}")
    for topic_id, words in topics_with_words:
        print(f"📝 Topic {topic_id}:")
        for word, weight in words:
            print(f"   - {word}: {weight:.4f}")
