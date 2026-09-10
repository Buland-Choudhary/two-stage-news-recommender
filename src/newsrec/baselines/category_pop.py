"""C1: category popularity with the user's dominant history category first."""
import numpy as np


def category_rankings(news, impressions, ids, histories, pools):
    categories = news.set_index('news_id').reindex(ids).category.fillna('unknown').astype(str)
    category_names, item_category = np.unique(categories, return_inverse=True)
    clicks = impressions.loc[(impressions.split == 'train') & (impressions.label == 1), 'news_id']
    item_index = {item: i for i, item in enumerate(ids)}
    counts = np.bincount([item_category[item_index[str(item)]] for item in clicks], minlength=len(category_names))
    preferences = []
    for history in histories:
        valid = history[history >= 0]
        preferences.append(int(np.bincount(item_category[valid], minlength=len(category_names)).argmax()))
    rankings = {}
    for name, pool in {'full': np.arange(len(ids)), **pools}.items():
        by_category = []
        for category in range(len(category_names)):
            # Category counts transfer to unseen articles. No item clicks or recency proxy.
            priority = (item_category[pool] == category).astype(np.int64)
            order = np.lexsort((pool, -counts[item_category[pool]], -priority))
            by_category.append(pool[order[:200]])
        rankings[name] = np.asarray(by_category)[preferences]
    return rankings
