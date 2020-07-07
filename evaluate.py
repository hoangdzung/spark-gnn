import os 
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from scipy.sparse import csr_matrix, lil_matrix

import torch 
import torch.nn as nn 

BATCH_SIZE = 128
BS = 1024

def read_node_label(filename,embedding_dict=None):
    fin = open(filename, 'r')
    X = []
    Y = []
    while 1:
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        if embedding_dict is None or vec[0] in embedding_dict:
            X.append(vec[0])
            Y.append(vec[1:])
    fin.close()
    return X, Y

class LogisticRegressionPytorch(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegressionPytorch, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
        self.num_classes = num_classes
        # self.loss_fn = nn.BCEWithLogitsLoss()
        self.loss_fn = nn.MultiLabelSoftMarginLoss()
    
    def forward(self, x):
        out = self.linear(x)
        return out

    def loss(self, x, y):
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        return loss
    
    def predict(self, x, top_k_list):
        predicts = lil_matrix((x.shape[0], self.num_classes))
        n_iter = len(x) // BS + 1
        x = torch.FloatTensor(x)
        for iter in range(n_iter):
            batchX = x[iter*BS:(iter+1)*BS].cuda()
            scores_batch = self.forward(batchX)
            scores_batch = scores_batch.argsort(dim=1, descending=True).detach().cpu().numpy()
            scores = lil_matrix(scores_batch.shape, dtype=np.int32)
            for i in range(len(batchX)):
                scores[i, scores_batch[i, :top_k_list[i]]] = 1
            top_k_list = top_k_list[BS:]
            predicts[iter*BS:(iter+1)*BS] = scores
            if iter % 100 == 0:
                print('Predict iter {}/{}'.format(iter, n_iter))
        return predicts

class ClassifierPytorch(object):

   def __init__(self, vectors):
      self.embeddings = vectors
      self.binarizer = MultiLabelBinarizer(sparse_output=True)
      self.clf = None

   def train(self, X, Y, Y_all):
      print('Transforming labels')
      self.binarizer.fit(Y_all)
      print('Done transform labels')
      input_size = len(list(self.embeddings.values())[0])
      self.clf = LogisticRegressionPytorch(input_size, len(self.binarizer.classes_)).cuda()
      learning_rate = 0.1
      n_iter = len(X) // BATCH_SIZE
      optimizer = torch.optim.Adam(self.clf.parameters(), lr=learning_rate)  
      X = torch.FloatTensor([self.embeddings[x] for x in X])
      total_iter = 0
      for epoch in range(50):
         for iter in range(n_iter):
               batchX = X[iter*BATCH_SIZE:(iter+1)*BATCH_SIZE].cuda()
               batchY = torch.FloatTensor(self.binarizer.transform(
                  Y[iter*BATCH_SIZE:(iter+1)*BATCH_SIZE]).todense()).cuda()
               optimizer.zero_grad()
               loss = self.clf.loss(batchX, batchY)
               loss.backward()
               if iter % 1000 == 0:
                  print('Epoch: %d - Iter: %d/%d, Loss: %.4f' % (epoch, iter, n_iter, loss.item()))
               optimizer.step()
               total_iter += 1
               if total_iter > 10000:
                  break
         if total_iter > 10000:
               break
      # Y = self.binarizer.transform(Y)
      # self.clf.fit(X_train, Y)

   def evaluate(self, X, Y):
      top_k_list = [len(l) for l in Y]
      Y_ = self.predict(X, top_k_list)
      Y = self.binarizer.transform(Y)
      averages = ["micro", "macro", "samples", "weighted"]
      results = {}
      for average in averages:
         results[average] = f1_score(Y, Y_, average=average)
      # print('Results, using embeddings of dimensionality', len(self.embeddings[X[0]]))
      # print('-------------------')
      # print(results)
      return results
      # print('-------------------')

   def predict(self, X, top_k_list):
      X_ = np.asarray([self.embeddings[x] for x in X])
      Y = self.clf.predict(X_, top_k_list=top_k_list)
      return Y

   def split_train_evaluate(self, X, Y, train_percent, seed=0):
      state = np.random.get_state()

      training_size = int(train_percent * len(X))
      np.random.seed(seed)
      shuffle_indices = np.random.permutation(np.arange(len(X)))
      X_train = [X[shuffle_indices[i]] for i in range(training_size)]
      Y_train = [Y[shuffle_indices[i]] for i in range(training_size)]
      X_test = [X[shuffle_indices[i]] for i in range(training_size, len(X))]
      Y_test = [Y[shuffle_indices[i]] for i in range(training_size, len(X))]

      self.train(X_train, Y_train, Y)
      np.random.set_state(state)
      return self.evaluate(X_test, Y_test)

def eval_classify(embedding_dict, label_file, train_percent=0.5, seed=123):
    # label_file = os.path.join(graphdir,'labels.txt')
    if not os.path.isfile(label_file):
        label_file = os.path.join(label_file,'labels.txt')
    if not os.path.isfile(label_file):
        print('Label file does not exist')
        return {}
    clf = ClassifierPytorch(vectors=embedding_dict)
    X,Y = read_node_label(label_file,embedding_dict)
    scores = clf.split_train_evaluate(X, Y, train_percent=train_percent, seed=seed)
    return scores