import nltk

def caption_len(caption):
    tokens = nltk.tokenize.word_tokenize(caption.lower())
    return len(tokens)

def get_batches(captions_len, batch_size):
    d = {}
    for idx, cap_len in enumerate(captions_len):
        if cap_len in d:
            d[cap_len].append(idx)
        else:
            d[cap_len] = [idx]
    batches = []
    cur_batch = []
    for indices in d.values():
        for idx in indices:
            cur_batch.append(idx)
            if len(cur_batch) == batch_size:
                batches.append(cur_batch)
                cur_batch = []
        if cur_batch:
            batches.append(cur_batch)
            cur_batch = []
    return batches