# %%
import spacy
import json
import numpy as np
from typing import Dict, List
from collections import Counter
from preprocessing_1 import tokenize_text, load_spacy_model
from sklearn.metrics.pairwise import cosine_similarity

QUERY = "highest pandemic casualties"

CORPUS_PATH = "data/corpus_tfidf.json"
CORPUS_VOCAB_PATH = "data/corpus_vocab.json"
IDF_DICT_PATH = "data/idf_values.json"


def load_file(filepath: str) -> List:
    with open(filepath, "r") as f:
        data = json.load(f)
    return data


def vectorize_query(
    query: str,
    spacy_model: spacy.language.Language,
    corpus_vocabulary: List,
    idf_dict: Dict,
) -> np.array:
    query_tokenized = tokenize_text(query, spacy_model)
    query_token_counts = Counter(query_tokenized)
    total_query_tokens = len(query_tokenized)

    idf_values = [idf_dict[token] for token in corpus_vocabulary]

    query_term_frequency = np.array(
        [query_token_counts[token] / total_query_tokens for token in corpus_vocabulary]
    )

    return query_term_frequency * idf_values


def search_tfidf(corpus: List[Dict], query_tfidf: np.array) -> List[Dict]:
    rankings = []
    for doc in corpus:
        doc_rank = {}
        doc_tfidf = np.array(doc["tf_idf_vector"])
        ranking = cosine_similarity(
            query_tfidf.reshape(1, -1), doc_tfidf.reshape(1, -1)
        )[0][0]
        if ranking > 0:
            doc_rank["title"] = doc["title"]
            doc_rank["rank"] = ranking
            rankings.append(doc_rank)

    return sorted(rankings, key=lambda x: x["rank"], reverse=True)


def main():
    nlp = load_spacy_model()

    corpus = load_file(CORPUS_PATH)
    corpus_vocabulary = load_file(CORPUS_VOCAB_PATH)
    idf_dict = load_file(IDF_DICT_PATH)

    query_tfidf = vectorize_query(QUERY, nlp, corpus_vocabulary, idf_dict)

    print(search_tfidf(corpus, query_tfidf))


if __name__ == "__main__":
    main()
