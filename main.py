import sys
import os
import argparse

import networkx as nx 
import numpy as np 

from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext

from hdfs import InsecureClient

from deepwalk import DeepWalk,embed_by_deepwalk
from utils import model2dict
from evaluate import eval_classify

def parse_args():
   parser = argparse.ArgumentParser(description="Parallel embedding")
   parser.add_argument('--data_dir', default="./cora")
   parser.add_argument('--label_file', default="./cora")

   parser.add_argument('--num_subgraphs', default=3,
                     type=int, help="Number of subgraphs")
   parser.add_argument('--dim_size', default=128,
                     type=int, help="Dimension size")
   parser.add_argument('--seed', default=42, type=int)

   parser.add_argument('--train_percent', default=0.5,
                     type=float, help="Train percent")
    
   # Embed by deepwalk
   parser.add_argument('--walk_length', default=10,
                     type=int, help="Walk length")
   parser.add_argument('--num_walks', default=20,
                     type=int, help="Number of walks")
   parser.add_argument('--num_workers', default=4, type=int,
                     help="Number of workers for gensim")
   parser.add_argument('--window_size', default=5,
                     type=int, help="Window size")
   return parser.parse_args()

args = parse_args()

spark = SparkSession.builder.appName("HDFS").getOrCreate()
sc = SparkContext.getOrCreate(SparkConf().setAppName("HDFS"))

# client_hdfs = InsecureClient('http://localhost:50070')
# graph_files = client_hdfs.list(data_dir) 

graph_files = os.listdir(args.data_dir)

assert 'anchors.edgelist' in graph_files
graph_files.remove('anchors.edgelist')
graph_files = [os.path.join(args.data_dir, i) for i in graph_files]


anchors_graph = nx.read_edgelist(os.path.join(args.data_dir, 'anchors.edgelist'))
anchors_model =  embed_by_deepwalk(anchors_graph, 
                  args.dim_size,
                  args.num_walks, 
                  args.walk_length,
                  args.num_workers)

anchors_model_dict = model2dict(anchors_model)

num_paths_bd = sc.broadcast(args.num_walks)
path_length_bd = sc.broadcast(args.walk_length)
dim_size_bd = sc.broadcast(args.dim_size)
window_size_bd = sc.broadcast(args.window_size)
num_workers_bd = sc.broadcast(args.num_workers)
anchors_model_dict_bd = sc.broadcast(anchors_model_dict)

def map_train_embedding(filepath):
    subgraph = nx.read_edgelist(filepath)
    model = embed_by_deepwalk(subgraph, 
                  dim_size_bd.value,
                  num_paths_bd.value, 
                  path_length_bd.value,
                  num_workers_bd.value)

    nodes = list(anchors_model_dict_bd.value.keys())
    anchors_emb = np.stack(list(anchors_model_dict_bd.value.values()))
    anchors_emb = anchors_emb/np.sqrt(np.sum(anchors_emb**2,axis=1,keepdims=True))

    embeddings = model.wv[nodes]
    embeddings = embeddings/np.sqrt(np.sum(embeddings**2,axis=1,keepdims=True))

    trans_matrix, _, _,_ = np.linalg.lstsq(anchors_emb, embeddings, rcond=-1)

    nodes = list(model.wv.vocab.keys())
    embeddings = model.wv[nodes]
    embeddings = embeddings/np.sqrt(np.sum(embeddings**2,axis=1,keepdims=True))

    new_embeddings = np.matmul(embeddings, trans_matrix)
    return list(zip(nodes, new_embeddings))

my_list_rdd = sc.parallelize(graph_files).flatMap(lambda x: map_train_embedding(x)).combineByKey(lambda x : x, lambda x, y: x, lambda x,y :x)
final_model = dict(my_list_rdd.collect())

sc.stop()
scores = eval_classify(final_model, args.label_file, args.train_percent, args.seed)
print(scores)


# my_list_rdd = sc.parallelize(graph_files).map(lambda x: map_train_embedding(x))
# final_model = my_list_rdd.collect()

