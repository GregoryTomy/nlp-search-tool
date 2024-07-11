# Building a Search Tool with Natural Language Processing

In this project, we build an efficient and smart search engine for fast access to the CDC’s document database. The goal is to support the development of a new plan by the Department of Health and Human Services to prevent, control, and respond to future pandemics. This involves researching and developing strategies to aggregate and quickly search historical and likely unstructured text data about earlier pandemics, ensuring they can be easily accessed when needed.

## Business Problem

The CDC holds vast amounts of unstructured text data on past pandemics, which is crucial for informing current and future public health strategies. However, efficiently searching and retrieving relevant information from this extensive database poses a significant challenge. An effective search engine is needed to enable quick access to vital documents, facilitating informed decision-making and timely responses to public health emergencies.

## [CURRENT] Semantic Search with BERT
In this stage, we use transfer learning with BERT to build a semantic search engine.

## Unstructured Text Search

• **TF-IDF Search**: Assessed the effectiveness of Term Frequency-Inverse Document Frequency (TF-IDF) for retrieving relevant documents from the database.
- Implemeted TF-IDF search algorithm using `en_core_web_sm` language model from `spaCy`. However, TF-IDF based search can be highly inefficient for very large documents sets since we need to compute the similarities between the query and each document in the database.

•**Inverted Index**: Implemented and evaluated an inverted index to facilitate fast text search.
- Search Result Example:

    Query: "economic impact of pandemics"

    | Document Title                              | TF-IDF Score |
    |---------------------------------------------|--------------|
    | Pandemic prevention                         | 0.08262      |
    | Pandemic Severity Assessment Framework      | 0.05147      |
    | Pandemic                                    | 0.03380      |
    | HIV/AIDS                                    | 0.02889      |
    | Pandemic severity index                     | 0.02741      |
    | COVID-19 pandemic                           | 0.02479      |
    | PREDICT (USAID)                             | 0.02023      |
    | Spanish flu                                 | 0.01902      |
    | Crimson Contagion                           | 0.01655      |
    | 1929–1930 psittacosis pandemic              | 0.01620      |
    | Plague of Cyprian                           | 0.01574      |
    | Swine influenza                             | 0.00674      |
    | Antonine Plague                             | 0.00649      |
    | Science diplomacy and pandemics             | 0.00452      |
    | Epidemiology of HIV/AIDS                    | 0.00292      |
    | Unified Victim Identification System        | 0.00285      |
    | Cholera                                     | 0.00179      |
