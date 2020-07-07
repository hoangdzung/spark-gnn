import random
import time
from collections import defaultdict
import multiprocessing

from tqdm import tqdm
import numpy as np
import networkx as nx
from gensim.models import Word2Vec
from openne import walker

class DeepWalk():
    def __init__(self,G, num_paths=10, path_length=40, representation_size=64, window=5, workers=1, alpha=0):
        self.G = G 
        self._build_adjlist()
        self._build_deepwalk_corpus(num_paths, path_length, alpha)
        self._train(representation_size, window, workers)

    def _build_adjlist(self):
        self.adj = defaultdict(list)
        for node in tqdm(self.G.nodes(),desc='Build adjlist'):
            self.adj[node] = list(self.G.neighbors(node))
             
    def _random_walk(self, path_length, alpha=0, rand=random.Random(), start=None):
        """ Returns a truncated random walk.
            path_length: Length of the random walk.
            alpha: probability of restarts.
            start: the start node of the random walk.
        """
        if start:
            path = [start]
        else:
            # Sampling is uniform w.r.t V, and not w.r.t E
            path = [rand.choice(list(self.G.nodes()))]

        while len(path) < path_length:
            cur = path[-1]
            if len(self.adj[cur]) > 0:
                if rand.random() >= alpha:
                    path.append(rand.choice(self.adj[cur]))
                else:
                    path.append(path[0])
            else:
                break
        return [str(node) for node in path]

    # TODO add build_walks in here

    def _build_deepwalk_corpus(self, num_paths, path_length, alpha=0,
                        rand=random.Random(0)):
        self.walks = []

        nodes = list(self.G.nodes())
        
        for _ in tqdm(range(num_paths), desc='Build deepwalk corpus'):
            rand.shuffle(nodes)
            for node in nodes:
                self.walks.append(self._random_walk(path_length, rand=rand, alpha=alpha, start=node))
    
    def _train(self, size, window, workers):
        print('Start training...')
        stime = time.time()
        self.model = Word2Vec(self.walks, size=size, window=window, min_count=0, sg=1, hs=1, workers=workers)
        print('Done training... ', time.time()-stime, 's')


class Graph(object):
    def __init__(self):
        self.G = None
        self.look_up_dict = {}
        self.look_back_list = []
        self.node_size = 0

    def encode_node(self):
        look_up = self.look_up_dict
        look_back = self.look_back_list
        for node in self.G.nodes():
            look_up[node] = self.node_size
            look_back.append(node)
            self.node_size += 1
            self.G.nodes[node]['status'] = ''

    def read_g(self, g):
        self.G = g
        self.encode_node()

    def read_adjlist(self, filename):
        """ Read graph from adjacency file in which the edge must be unweighted
            the format of each line: v1 n1 n2 n3 ... nk
            :param filename: the filename of input file
        """
        self.G = nx.read_adjlist(filename, create_using=nx.DiGraph())
        for i, j in self.G.edges():
            self.G[i][j]['weight'] = 1.0
        self.encode_node()

    def read_edgelist(self, filename, weighted=False, directed=False):
        self.G = nx.DiGraph()

        if directed:
            def read_unweighted(l):
                src, dst = l.split()
                self.G.add_edge(src, dst)
                self.G[src][dst]['weight'] = 1.0

            def read_weighted(l):
                src, dst, w = l.split()
                self.G.add_edge(src, dst)
                self.G[src][dst]['weight'] = float(w)
        else:
            def read_unweighted(l):
                src, dst = l.split()
                self.G.add_edge(src, dst)
                self.G.add_edge(dst, src)
                self.G[src][dst]['weight'] = 1.0
                self.G[dst][src]['weight'] = 1.0

            def read_weighted(l):
                src, dst, w = l.split()
                self.G.add_edge(src, dst)
                self.G.add_edge(dst, src)
                self.G[src][dst]['weight'] = float(w)
                self.G[dst][src]['weight'] = float(w)
        fin = open(filename, 'r')
        func = read_unweighted
        if weighted:
            func = read_weighted
        while 1:
            l = fin.readline()
            if l == '':
                break
            func(l)
        fin.close()
        self.encode_node()

    def read_node_label(self, filename):
        fin = open(filename, 'r')
        while 1:
            l = fin.readline()
            if l == '':
                break
            vec = l.split()
            self.G.nodes[vec[0]]['label'] = vec[1:]
        fin.close()

    def read_node_features(self, filename):
        fin = open(filename, 'r')
        for l in fin.readlines():
            vec = l.split()
            self.G.nodes[vec[0]]['feature'] = np.array(
                [float(x) for x in vec[1:]])
        fin.close()

    def read_node_status(self, filename):
        fin = open(filename, 'r')
        while 1:
            l = fin.readline()
            if l == '':
                break
            vec = l.split()
            self.G.nodes[vec[0]]['status'] = vec[1]  # train test valid
        fin.close()

    def read_edge_label(self, filename):
        fin = open(filename, 'r')
        while 1:
            l = fin.readline()
            if l == '':
                break
            vec = l.split()
            self.G[vec[0]][vec[1]]['label'] = vec[2:]
        fin.close()

def embed_by_deepwalk(subgraph, dim_size, number_walks, walk_length, workers, seed=42):
    graph = Graph()
    graph.read_g(subgraph)

    walk = walker.BasicWalker(graph, workers=workers)
    sentences = walk.simulate_walks2(
        num_walks=number_walks, walk_length=walk_length, num_workers=workers)

    for idx in range(len(sentences)):
        sentences[idx] = [str(x) for x in sentences[idx]]

    print("Learning representation...")

    word2vec = Word2Vec(sentences=sentences, min_count=0, workers=workers,
                            size=dim_size, sg=1, seed=seed)
    return word2vec

if __name__ == '__main__':
    from utils import load_data
    G = load_data('./cora')
    deepwalk = DeepWalk(G)
