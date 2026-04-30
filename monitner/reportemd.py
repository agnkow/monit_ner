# --------------- Embedding ---------------
from monitner.embdrift import extract_tokens_vectors, \
                            token_centroid_drift, \
                            entity_centroid_drift, \
                            drift_js, \
                            drift_kl, \
                            oov_rate


# Embedding report
def data_emb_report(
        df_a,
        df_b,
        nlp,
        col_id='id',
        col_text='text'
):

    # Feature extraction for two datasets
    df_token_a = extract_tokens_vectors(df_a[[col_id, col_text]], nlp)
    df_token_b = extract_tokens_vectors(df_b[[col_id, col_text]], nlp)
    df_token_a['token_lower'] = df_token_a['token'].str.lower()
    df_token_b['token_lower'] = df_token_b['token'].str.lower()
    print('\n')


    # --------------- Tokens ---------------
    # Token embedding centroid drift
    t_centroid_drift = token_centroid_drift(df_token_a, df_token_b)
    print("Token embedding centroid drift:", round(t_centroid_drift, 3))

    # OOV / new tokens
    print("OOV rate (tokeny):", oov_rate(df_token_a, df_token_b, 'token_lower'))


    # --------------- Entities ---------------
    # Only non-entity tokens
    df_token_a_0 = df_token_a[df_token_a.entity == 'O']
    df_token_b_0 = df_token_b[df_token_b.entity == 'O']

    # Only entity tokens
    df_entity_a = df_token_a[df_token_a.entity != 'O']
    df_entity_b = df_token_b[df_token_b.entity != 'O']

    # Embedding centroid drift
    non_entity_centroid_drift = token_centroid_drift(df_token_a_0, df_token_b_0)
    print("Non-entity token embedding centroid drift:", round(non_entity_centroid_drift, 3))

    all_entity_centroid_drift = token_centroid_drift(df_entity_a, df_entity_b)
    print("Entity embedding centroid drift:", round(all_entity_centroid_drift, 3))
    print('\n')

    # PER / ORG / LOC embedding drift
    for label in ['persName', 'orgName', 'placeName', 'geogName']:
        drift = entity_centroid_drift(df_entity_a, df_entity_b, label)
        print(f"{label} centroid drift:", round(drift, 3))
    print('\n')

    # Entity distribution drift: Jensen–Shannon
    js = drift_js(df_entity_a, df_entity_b)
    print("Entity distribution drift (JS):", round(js, 3))

    # Entity distribution drift: KL divergence
    kl = drift_kl(df_entity_a, df_entity_b)
    print("Entity distribution drift (KL):", round(kl, 3))
    print('\n')

    # OOV / new entities
    print("OOV rate (entities):", oov_rate(df_entity_a, df_entity_b, 'token_lower'))
    print('\n')
