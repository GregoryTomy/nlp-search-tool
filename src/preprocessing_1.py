# %%
import json
import spacy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_spacy_model(model_name: str = "en_core_web_sm") -> spacy.language.Language:
    """_summary_

    Args:
        model_name (str): _description_

    Returns:
        spacy.language.Language: _description_
    """
    if not spacy.util.is_package(model_name):
        logger.info(f"Model {model_name} not found. Downloading...")
        spacy.cli.download(model_name)
        logger.info(f"Model {model_name} downloaded successfully.")
    else:
        logger.info(f"Model {model_name} already downloaded.")

    nlp = spacy.load(model_name)

    return nlp


def load_data(path_to_data: str) -> list:
    """_summary_

    Args:
        path_to_data (str): _description_

    Returns:
        list: _description_
    """
    with open(path_to_data, "r") as file:
        data = json.load(file)

    return data


# %%
nlp = load_spacy_model()
data = load_data("../data/data.json")
print(data)

# %%
text = data[1]["text"]
tokenized_text = nlp(text.lower())

for token in tokenized_text[:5]:
    print(type(token), token.text, token.pos_, token.dep_)
# %%
