"""C2: lexical cosine retrieval from a mean TF-IDF history profile."""
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from newsrec.encoders import article_text


def stable_topk(scores, pool, k=200):
    """Descending score, corpus-index tie break, including ties at the K boundary."""
    k = min(k, len(pool))
    threshold = np.partition(scores, len(scores) - k)[len(scores) - k]
    above = np.flatnonzero(scores > threshold)
    tied = np.flatnonzero(scores == threshold)
    tied = tied[np.argsort(pool[tied])][:k - len(above)]
    head = np.concatenate([above, tied])
    return pool[head[np.lexsort((pool[head], -scores[head]))]]


def tfidf_rankings(news, impressions, ids, histories, pools):
    news = news.set_index('news_id').reindex(ids)
    texts = [article_text(row.title, row.abstract) for row in news.itertuples()]
    training = set(impressions.loc[impressions.split == 'train', 'news_id'].astype(str))
    vectorizer = TfidfVectorizer(max_features=50000, min_df=2, stop_words='english', dtype=np.float32)
    vectorizer.fit([text for item, text in zip(ids, texts, strict=True) if item in training])
    items = vectorizer.transform(texts)
    rows, cols = [], []
    for i, history in enumerate(histories):
        valid = history[history >= 0]
        rows.extend([i] * len(valid))
        cols.extend(valid.tolist())
    selector = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                                 shape=(len(histories), len(ids)))
    profiles = normalize(selector @ items)
    universes = {'full': np.arange(len(ids)), **pools}
    output = {name: np.empty((len(histories), min(200, len(pool))), dtype=np.int64)
              for name, pool in universes.items()}
    for start in range(0, len(histories), 128):
        scores = (profiles[start:start + 128] @ items.T).toarray()
        for j, row in enumerate(scores):
            for name, pool in universes.items():
                output[name][start + j] = stable_topk(row[pool], pool)
        if start % 8192 == 0:
            print(f'C2 scored {start}/{len(histories)} query profiles', flush=True)
    return output, dict(vocabulary_size=len(vectorizer.vocabulary_), fit_articles=len(training),
                        vocabulary_fit='training-window articles only', query_batch_size=128)
