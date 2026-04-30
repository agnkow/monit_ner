from collections import Counter

import numpy as np
import pandas as pd
import spacy


def properties_text(df, col_text, nlp_spacy):
    # spacy features - tokens, sentences
    df['tokens_sents_per_doc'] = df[col_text].apply(tokens_sents_per_doc, args=(nlp_spacy,))
    df_propert = pd.DataFrame(df['tokens_sents_per_doc'].tolist())
    df = pd.concat([df, df_propert], axis=1)
    df = df.drop(columns='tokens_sents_per_doc')
    return df


def tokens_sents_per_doc(text, nlp_spacy):
    doc = nlp_spacy(str(text))

    return {
        'tokens': tokens_per_doc(doc),
        'words': words_per_doc(doc),
        'entities': entities_per_doc(doc),
        'punctuation': punct_per_doc(doc),
        'digits': digits_per_doc(doc),
        'symbols': symbols_per_doc(doc),
        'capital': capital_per_doc(doc),
        'sentences': sents_per_doc(doc),
        'words_per_sent': words_per_sent(doc),
        'tokens_per_sent': tokens_per_sent(doc),
        'entities_per_sent': entities_per_sent(doc),
        'mean_length_word': mean_len_word_per_doc(doc),
        'pos': pos_per_doc(doc),
        'pos_distrib': pos_distrib_per_doc(doc),
        'mean_length_ent': mean_len_ent_per_doc(doc)
    }


def tokens_per_doc(doc):
    return len(doc)


def words_per_doc(doc):
    words = 0
    for token in doc:
        if token.is_alpha:
            words += 1
    return words


def entities_per_doc(doc):
    return len(doc.ents)


def punct_per_doc(doc):
    punct = 0
    for token in doc:
        if token.is_punct:
            punct += 1
    return punct


def digits_per_doc(doc):
    digits = 0
    for token in doc:
        if token.is_digit:
            digits += 1
    return digits


def symbols_per_doc(doc):
    symbols = 0
    for token in doc:
        if token.pos_ == "SYM":
            symbols += 1
    return symbols


def capital_per_doc(doc):
    capital = 0
    for token in doc:
        if token.is_title:
            capital += 1
    return capital


def sents_per_doc(doc):
    return len(list(doc.sents))


def words_per_sent(doc):
    words_list = []
    for sent in doc.sents:
        words = 0
        for token in sent:
            if token.is_alpha:
                words += 1
        words_list.append(words)
    return words_list


def tokens_per_sent(doc):
    tokens_list = []
    for sent in doc.sents:
        tokens = 0
        for token in sent:
            if not token.is_punct:
                tokens += 1
        tokens_list.append(tokens)
    return tokens_list


def entities_per_sent(doc):
    entities_list = []
    for sent in doc.sents:
        entities = 0
        tokens = 0
        for token in sent:
            if token.ent_type_:
                entities += 1
        entities_list.append(entities)
    return entities_list


def mean_len_word_per_doc(doc):
    length_words_list = []
    for token in doc:
        if token.is_alpha:
            length_words_list.append(len(token.text))
    return np.mean(length_words_list).round(2)


def pos_per_doc(doc):
    pos = []
    for token in doc:
        if not token.is_punct:
            pos.append(token.pos_)
    return dict(Counter(pos))


def pos_distrib_per_doc(doc):
    pos_counts = pos_per_doc(doc)
    total = sum(pos_counts.values())
    return {pos: count / total for pos, count in pos_counts.items()}


def mean_len_ent_per_doc(doc):
    lengths = []
    for ent in doc.ents:
        lengths.append(len(ent.text))
    return np.mean(lengths).round(2)
