def model2dict(model):
    nodes = list(model.wv.vocab.keys())
    embeddings = model.wv[nodes]

    return dict(zip(nodes, embeddings))