import itertools
import json
from typing import Dict, List, Tuple
import numpy as np
from collections import Counter
from preprocessing_1 import load_data

DATA_PATH = "data/preprocessed.json"  # preprocessed data from 1
CORPUS_VOCAB_PATH = "data/corpus_vocab.json"
TF_IDF_PATH = "data/tf_idf_vectors.json"


def load_and_process_data(filepath: str) -> Tuple[List[Dict], List[List]]:
    """_summary_

    Args:
        filepath (str): _description_

    Returns:
        Tuple[List[Dict], List[List]]: _description_
    """
    corpus = load_data(filepath)
    token_list = [doc["tokenized_text"] for doc in corpus]

    return corpus, token_list


def get_corpus_vocabulary(token_list: List[List]) -> List:
    corpus_vocabulary = [list for sublist in token_list for list in sublist]
    return list(set(corpus_vocabulary))


def save_to_file(data: List, filepath: str) -> None:
    with open(filepath, "w") as f:
        json.dump(data, f)


def get_document_frequencies(token_list: List[List]) -> Counter:
    return Counter(itertools.chain(*[set(doc) for doc in token_list]))


def get_inverse_document_frequency(
    corpus_size: int, document_frequencies: Counter, corpus_vocabulary: List
):
    return {
        token: np.log(corpus_size / document_frequencies[token])
        for token in corpus_vocabulary
    }


def calculate_tf_idf_vector(
    token_list: List, idf_dict: Dict, corpus_vocabulary: List
) -> List[np.array]:
    token_counter = Counter(token_list)
    total_tokens = len(token_list)

    term_frequency = np.array(
        [token_counter[token] / total_tokens for token in corpus_vocabulary]
    )
    idf_values = np.array([idf_dict[token] for token in corpus_vocabulary])

    return term_frequency * idf_values


def main():
    corpus, token_lists = load_and_process_data(DATA_PATH)

    corpus_vocabulary = get_corpus_vocabulary(token_lists)

    save_to_file(corpus_vocabulary, CORPUS_VOCAB_PATH)

    document_frequencies = get_document_frequencies(token_lists)

    inverse_document_frequency_dict = get_inverse_document_frequency(
        len(corpus), document_frequencies, corpus_vocabulary
    )

    # TODO: return original corpus with tf_idf_vectors as additional field
    tf_idf_vectors = []
    for token_list in token_lists:
        tf_idf_vector = calculate_tf_idf_vector(
            token_list, inverse_document_frequency_dict, corpus_vocabulary
        )
        tf_idf_vectors.append(list(tf_idf_vector))

    save_to_file(tf_idf_vectors, TF_IDF_PATH)


if __name__ == "__main__":
    main()
# %%
