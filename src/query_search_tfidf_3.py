import spacy
import json
import logging
import numpy as np
from typing import Dict, List, Union
from collections import Counter
from preprocessing_1 import tokenize_text, load_spacy_model
from sklearn.metrics.pairwise import cosine_similarity

QUERY = "highest pandemic casualties"

CORPUS_PATH = "data/corpus_tfidf.json"
CORPUS_VOCAB_PATH = "data/corpus_vocab.json"
IDF_DICT_PATH = "data/idf_values.json"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_file(filepath: str) -> List[Union[List, str, Dict]]:
    """Load file from a given path.

    Args:
        filepath (str): Path to the file to load.

    Returns:
        List[Union[List, str, Dict]]: Data loaded from JSON file
    """
    with open(filepath, "r") as f:
        data = json.load(f)
    return data


def vectorize_query(
    query: str,
    spacy_model: spacy.language.Language,
    corpus_vocabulary: List,
    idf_dict: Dict,
) -> np.array:
    """Vectorize a query using the provided spaCy model, corpus vocabulary, and IDF values.

    Args:
        query (str): Query sting to vectorize.
        spacy_model (spacy.language.Language): Loaded spaCy model for tokenization.
        corpus_vocabulary (List): Corpus vocabulary used for TF-IDF of the corpus.
        idf_dict (Dict): IDF values for tokens in the corpus vocabulary.

    Returns:
        np.array: TF-IDF vector for the query.
    """
    logger.info("Tokenizing query.")
    query_tokenized = tokenize_text(query, spacy_model)
    query_token_counts = Counter(query_tokenized)
    total_query_tokens = len(query_tokenized)

    # Ensure idf values are in the same order as the tokens in the corpus vocabulary.
    idf_values = np.array([idf_dict[token] for token in corpus_vocabulary])

    query_term_frequency = np.array(
        [query_token_counts[token] / total_query_tokens for token in corpus_vocabulary]
    )
    logger.info("Query vectorized successfully.")

    return query_term_frequency * idf_values


def search_tfidf(corpus: List[Dict], query_tfidf: np.array) -> List[Dict]:
    """Search the corpus for documents similar to the query using TF-IDF vectors.

    Args:
        corpus (List[Dict]): Corpus of documents with TF-IDF vectors.
        query_tfidf (np.array): TF-IDF vector for the query.

    Returns:
        List[Dict]: A list of documents ranked by their similarity to the query.
    """
    logger.info("Calculating cosine similarity.")
    doc_tfidf_matrix = np.array([doc["tf_idf_vector"] for doc in corpus])
    cosine_similarities = cosine_similarity(
        query_tfidf.reshape(1, -1), doc_tfidf_matrix
    )[0]

    #! Currently returns all matches with rankings greater than 0.
    rankings = [
        {"title": doc["title"], "rank": similarity}
        for doc, similarity in zip(corpus, cosine_similarities)
        if similarity > 0
    ]
    logger.info("Cosine similarity calculation completed.")
    return sorted(rankings, key=lambda x: x["rank"], reverse=True)


def main():
    nlp = load_spacy_model()

    corpus = load_file(CORPUS_PATH)
    corpus_vocabulary = load_file(CORPUS_VOCAB_PATH)
    idf_dict = load_file(IDF_DICT_PATH)

    query_tfidf = vectorize_query(QUERY, nlp, corpus_vocabulary, idf_dict)

    logger.info("Searching TF-IDF")
    results = search_tfidf(corpus, query_tfidf)

    for result in results:
        print(result)


if __name__ == "__main__":
    main()
