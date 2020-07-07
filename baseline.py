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

graph = nx.read_edgelist(os.path.join(args.data_dir, 'edgelist.txt'))
model =  embed_by_deepwalk(graph, 
                  args.dim_size,
                  args.num_walks, 
                  args.walk_length,
                  args.num_workers)

model_dict = model2dict(model)

scores = eval_classify(model_dict, args.label_file, args.train_percent, args.seed)
print(scores)


# my_list_rdd = sc.parallelize(graph_files).map(lambda x: map_train_embedding(x))
# final_model = my_list_rdd.collect()

