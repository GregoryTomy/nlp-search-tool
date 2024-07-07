import itertools
import json
import logging
from typing import Dict, List, Tuple
import numpy as np
from collections import Counter
from preprocessing_1 import load_data

DATA_PATH = "data/preprocessed.json"  # preprocessed data from 1
CORPUS_VOCAB_PATH = "data/corpus_vocab.json"
TF_IDF_PATH = "data/corpus_tfidf.json"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_process_data(filepath: str) -> Tuple[List[Dict], List[List]]:
    """Load and process corpus data from JSON file.

    Args:
        filepath (str): File path to preprocessed JSON data file.

    Returns:
        Tuple[List[Dict], List[List]]: A tupe containting the loaded corpus and a list of tokenized text.
    """
    logger.info(f"Loading data from {filepath}")
    corpus = load_data(filepath)
    token_lists = [doc["tokenized_text"] for doc in corpus]
    logger.info("Data loaded and processed successfully")
    return corpus, token_lists


def get_corpus_vocabulary(token_lists: List[List]) -> List:
    """Extract unique vocabulary from the entire corpus

    Args:
        token_lists (List[List]): List of tokenized texts.

    Returns:
        List: A list containing the unique vocabulary from the corpus.
    """
    corpus_vocabulary = [list for sublist in token_lists for list in sublist]
    return list(set(corpus_vocabulary))


def save_to_file(data: List, filepath: str) -> None:
    """Save data to JSON file.

    Args:
        data (List): Data to be saved.
        filepath (str): Path to save the JSON file.
    """
    logger.info(f"Saving data to {filepath}")
    with open(filepath, "w") as f:
        json.dump(data, f)
    logger.info("Data saved successfully")


def get_document_frequencies(token_lists: List[List]) -> Counter:
    """Calculate the number of documents each token is present in.

    Args:
        token_lists (List[List]): List of tokenized texts.

    Returns:
        Counter: _description_
    """
    logger.info("Calculating document frequencies for each token.")
    return Counter(itertools.chain(*[set(doc) for doc in token_lists]))


def get_inverse_document_frequency(
    corpus_size: int, document_frequencies: Counter, corpus_vocabulary: List
) -> Dict[str, float]:
    """Calculate the inverse document frequency (IDF) for each token in the vocabulary.

    Args:
        corpus_size (int): Total number of documents in the corpus.
        document_frequencies (Counter): Document frequencies of tokens.
        corpus_vocabulary (List): List of unique tokens in the corpus.

    Returns:
        Dict[str, float]: A dictionary with tokens as keys and their IDF as values.
    """
    logger.info("Calculating inverse document frequency for each token.")
    return {
        token: np.log(corpus_size / document_frequencies[token])
        for token in corpus_vocabulary
    }


def calculate_tf_idf_vector(
    token_list: List, idf_dict: Dict, corpus_vocabulary: List
) -> np.array:
    """Calculate the TF-IDF vector for a single document.

    Args:
        token_list (List): Tokenized text for the document.
        idf_dict (Dict): Dictionary of IDF values.
        corpus_vocabulary (List): List of unique tokens in the corpus.

    Returns:
        np.array: TF-IDF array vector for the document.
    """
    token_counter = Counter(token_list)
    total_tokens = len(token_list)
    logger.debug(
        f"Calcualting TF-IDF vector for a document with {total_tokens} tokens."
    )
    term_frequency = np.array(
        [token_counter[token] / total_tokens for token in corpus_vocabulary]
    )
    idf_values = np.array([idf_dict[token] for token in corpus_vocabulary])

    return term_frequency * idf_values


def main():
    logger.info("Starting TF-IDF vector computation process...")
    corpus, token_lists = load_and_process_data(DATA_PATH)

    corpus_vocabulary = get_corpus_vocabulary(token_lists)

    save_to_file(corpus_vocabulary, CORPUS_VOCAB_PATH)

    document_frequencies = get_document_frequencies(token_lists)

    inverse_document_frequency_dict = get_inverse_document_frequency(
        len(corpus), document_frequencies, corpus_vocabulary
    )

    for index, token_list in enumerate(token_lists):
        tf_idf_vector = calculate_tf_idf_vector(
            token_list, inverse_document_frequency_dict, corpus_vocabulary
        )
        corpus[index]["tf_idf_vector"] = list(tf_idf_vector)

    save_to_file(corpus, TF_IDF_PATH)

    logger.info("TF-IDF vector computation process completed successfully.")


if __name__ == "__main__":
    main()
