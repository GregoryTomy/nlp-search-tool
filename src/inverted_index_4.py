# %%
import pprint
import logging
from collections import defaultdict
from typing import Dict, List, Tuple
from preprocessing_1 import load_spacy_model, tokenize_text
from query_search_tfidf_3 import load_file

CORPUS_VOCAB_PATH = "data/corpus_vocab.json"
CORPUS_PATH = "data/corpus_tfidf.json"
EXAMPLE_QUERIES = "data/example_queries.json"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_inverted_index(
    corpus_vocabulary: List[str], corpus: List[Dict]
) -> Dict[str, Tuple[str, float]]:
    """Create an inverted index for a given corpus

    Args:
        corpus_vocabulary (List[str]): Corpus vocabulary.
        corpus (List[Dict]): Corpus.

    Returns:
        Dict[Tuple[str, float]]: A dictionary with tokens as keys and a list of tuples (document title, TF-IDF score)  for each token.
    """
    logger.info("Creating inverted index")
    inverted_index = {}

    for doc in corpus:
        title = doc.get("title")
        tf_idf_vector = doc.get("tf_idf_vector")

        for index, tf_idf_score in enumerate(tf_idf_vector):
            if tf_idf_score != 0:
                token = corpus_vocabulary[index]
                if token not in inverted_index:
                    inverted_index[token] = []
                inverted_index[token].append((title, tf_idf_score))

    logger.info("Inverted index successfully created.")
    return inverted_index


def search_inverted_index(
    query_tokenized: List[str], inverted_index: Dict[str, Tuple[str, float]]
) -> List[Tuple[str, float]]:
    """Search for tokenized terms in the inverted index and return document titles with TF-IDF scores.

    Args:
        query_tokenized (List[str]): List of tokenized query terms.
        inverted_index (Dict[str, Tuple[str, float]]): Inverted index.

    Returns:
        List[Tuple[str, float]]: List of tuples (document titles, cumulative TF-IDF score) sorted by TF-IDF score in descending order.
    """
    logger.info("Searching query in inverted index.")
    search = []
    for token in query_tokenized:
        search.extend(inverted_index[token])

    # create a list of compound TF-IDF scores.
    results = defaultdict(int)
    for title, tf_idf_score in search:
        results[title] += tf_idf_score

    # return sorted results
    results = [(title, tf_idf_score) for title, tf_idf_score in results.items()]

    logger.info(f"Search completed. Found {len(results)} matches.")

    return sorted(results, key=lambda x: x[1], reverse=True)


def main():
    nlp = load_spacy_model()
    corpus_vocabulary = load_file(CORPUS_VOCAB_PATH)
    corpus = load_file(CORPUS_PATH)

    inverted_index = create_inverted_index(corpus_vocabulary, corpus)

    example_queries = load_file(EXAMPLE_QUERIES)

    for query in example_queries:
        tokenized_query = tokenize_text(query, nlp)
        results = search_inverted_index(tokenized_query, inverted_index)

        pprint.pprint(f"Query: {query}")
        pprint.pprint(results)
        print("\n")


if __name__ == "__main__":
    main()
