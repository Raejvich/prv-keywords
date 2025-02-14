from src.data.load_application import read_application
from src.data.clean import clean_text
from src.data.lda_functions import find_optimal_topics_from_text


def main():
    file_path = r"C:\Users\Gustav\Desktop\exjobb\fulltext_en\SE539880C2.xml"
    # load text from xml file
    raw_text = read_application(file_path)

    # clean and tokenize text
    cleaned_text = clean_text(raw_text)
    print(cleaned_text)
    # LDA
    optimal_num_topics, lda_model, topics_with_words = find_optimal_topics_from_text(
        cleaned_text
    )

    # Print the results
    print(f"Optimal number of topics: {optimal_num_topics}")
    for topic_id, words in topics_with_words:
        print(f"📝 Topic {topic_id}:")
        for word, weight in words:
            print(f"   - {word}: {weight:.4f}")


if __name__ == "__main__":
    main()
