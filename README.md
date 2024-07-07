# Building a Search Tool with Natural Language Processing

In this project, we build an efficient and smart search engine for fast access to the CDC’s document database. The goal is to support the development of a new plan by the Department of Health and Human Services to prevent, control, and respond to future pandemics. This involves researching and developing strategies to aggregate and quickly search historical and likely unstructured text data about earlier pandemics, ensuring they can be easily accessed when needed.

## Business Problem

The CDC holds vast amounts of unstructured text data on past pandemics, which is crucial for informing current and future public health strategies. However, efficiently searching and retrieving relevant information from this extensive database poses a significant challenge. An effective search engine is needed to enable quick access to vital documents, facilitating informed decision-making and timely responses to public health emergencies.

## Part I: Unstructured Text Search

In this phase, we will explore essential methods for unstructured text search, including:

• **TF-IDF Search**: Assessed the effectiveness of Term Frequency-Inverse Document Frequency (TF-IDF) for retrieving relevant documents from the database.
- Implemeted TF-IDF search algorithm using `en_core_web_sm` language model from `spaCy`. However, TF-IDF based search can be highly inefficient for very large documents sets since we need to compute the similarities between the query and each document in the database.

•**Inverted Index**: Implementing and evaluating an inverted index to facilitate fast text search.