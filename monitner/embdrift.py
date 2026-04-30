import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


#-----------------    Embeddings    -----------------#
def extract_tokens_vectors(
        df_input
        , nlp
        , col_id = 'id'
        , col_text = 'text'
):

    ids_list = []
    tokens_list = []
    entities_list = []
    vectors_list = []

    texts = df_input[col_text].tolist()
    ids = df_input[col_id].tolist()

    for doc, text_id in tqdm(
        zip(nlp.pipe(texts, batch_size=32), ids),
        total=len(texts)
    ):
        for token in doc:
            ids_list.append(text_id)
            tokens_list.append(token.text)
            entities_list.append(token.ent_type_ or 'O')
            vectors_list.append(token.vector)

    df_output = pd.DataFrame({
        'id_text': ids_list,
        'token': tokens_list,
        'entity': entities_list,
        'vector': vectors_list
    })

    return df_output


#----------------    Centroid drift    -----------------#
def centroid_vecs(vectors):
    if len(vectors) == 0:
        return None

    vecs = np.vstack(vectors)
    vecs = vecs[np.linalg.norm(vecs, axis=1) > 0]  # usuwam zerowe wektory

    if vecs.shape[0] == 0:
        return None
    return vecs.mean(axis=0)


def centroid_drift(a, b):
    ca = centroid_vecs(a)
    cb = centroid_vecs(b)

    if ca is None or cb is None:
        return np.nan
    return 1 - cosine_similarity([ca],[cb])[0][0]


# tokens
def token_centroid_drift(df_a, df_b):
    return centroid_drift(df_a.vector.values, df_b.vector.values)


# entities
def entity_centroid_drift(df_a, df_b, label):
    vecs_a = df_a[df_a.entity == label].vector.values
    vecs_b = df_b[df_b.entity == label].vector.values

    if len(vecs_a) == 0 or len(vecs_b) == 0:
        return np.nan
    return centroid_drift(vecs_a, vecs_b)


#----------------   Distribution drift    -----------------#
def entity_distribution(df_a, df_b):
    distrib_a = df_a.entity.value_counts(normalize=True)
    distrib_b = df_b.entity.value_counts(normalize=True)

    all_entities = distrib_a.index.union(distrib_b.index)

    distrib_a = distrib_a.reindex(all_entities, fill_value=0)
    distrib_b = distrib_b.reindex(all_entities, fill_value=0)
    return distrib_a, distrib_b


# Entity distribution drift (Jensen–Shannon)
# → change in entity type
def drift_js(df_a, df_b):
    distrib_a, distrib_b = entity_distribution(df_a, df_b)
    return jensenshannon(distrib_a.values, distrib_b.values)


# KL divergence
def kl_divergence(a, b, epsilon=1e-12):
    a = np.asarray(a) + epsilon
    b = np.asarray(b) + epsilon

    a = a / a.sum()
    b = b / b.sum()
    return np.sum(a * np.log(a / b))


def drift_kl(df_a, df_b):
    distrib_a, distrib_b = entity_distribution(df_a, df_b)
    return kl_divergence(distrib_a.values, distrib_b.values)


#----------    OOV / nowe tokeny (lub encje)    -----------#
def oov_rate(df_a, df_b, token_lower):
    vocab_a = set(df_a[token_lower])
    vocab_b  = set(df_b[token_lower])
    new_tokens = vocab_b - vocab_a
    return len(new_tokens) / len(vocab_b)
