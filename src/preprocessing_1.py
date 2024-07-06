# %%
import json
from typing import List
import spacy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_PATH = "data/data.json"


def load_spacy_model(model_name: str = "en_core_web_sm") -> spacy.language.Language:
    """Load a spaCy language model, downloading it if necessary.

    Args:
        model_name (str): Name of the spaCy model to load.

    Returns:
        spacy.language.Language: The loaded spaCy language model.
    """
    if not spacy.util.is_package(model_name):
        logger.info(f"Model {model_name} not found. Downloading...")
        spacy.cli.download(model_name)
        logger.info(f"Model {model_name} downloaded successfully.")
    else:
        logger.info(f"Model {model_name} already downloaded.")

    nlp = spacy.load(model_name)

    return nlp


def load_data(path_to_data: str) -> List[dict]:
    """Load JSON data from file path.

    Args:
        path_to_data (str): The path to the JSON file containing the data.

    Returns:
        List[dict]: A list of dictionaries containting the loaded data.
    """
    logger.info(f"Loading data from {path_to_data}")
    try:
        with open(path_to_data, "r") as file:
            data = json.load(file)
        logger.info(f"Data loaded successfully from {path_to_data}")
    except Exception as e:
        logger.error(f"Failed to load data from {path_to_data}: {e}")
        raise

    return data


def tokenize_text(text: str, spacy_model: spacy.language.Language) -> List[str]:
    """Tokenize and lemmatize a given text using spaCy model.

    Args:
        text (str): Text to be tokenized and lemmatized.
        spacy_model (spacy.language.Language): SpaCy model to use.

    Returns:
        List[str]: A list of lemmas from the tokenized text, excluding stopwords and punctuation.
    """
    logger.info(f"Tokenizing text: {text[:30]}...")
    tokens = spacy_model(text.lower())
    # remove stopwords and punctuation
    tokens_without_sw = [
        token for token in tokens if not token.is_stop and not token.is_punct
    ]
    lemmas = [token.lemma_ for token in tokens_without_sw if token.dep_]

    logger.info(f"Tokenization complete. Found {len(lemmas)} tokens.")

    return lemmas


def tokenize_documents(
    documents: List[dict], spacy_model: spacy.language.Language
) -> List[dict]:
    """Tokenize and lemmatize a corpus of documents.

    Args:
        documents (List[dict]): A list of dictionaries, each containing a 'text' field to tokenize.
        spacy_model (spacy.language.Language): SpaCy language model to use.

    Returns:
        List[dict]: The input list of dictionaries with an added 'tokenized_text' field with tokens.
    """
    for document in documents:
        document["tokenized_text"] = tokenize_text(document["text"], spacy_model)

    return documents


if __name__ == "__main__":
    logger.info("Starting preprocessing main execution...")
    nlp = load_spacy_model()
    documents = load_data(path_to_data=DATA_PATH)

    logger.info(f"Preprocessing {len(documents)} documents.")
    documents_preprocessed = tokenize_documents(documents, nlp)

    logger.debug(documents_preprocessed[:2])
    logger.info("Document preprocessing completed.")
