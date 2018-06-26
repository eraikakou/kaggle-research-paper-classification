import networkx as nx
import numpy as np
import pandas as pd
import os


class GraphFeatures(object):
	
	GRAPH_DATA_PATH = "./data/Cit-HepTh.txt"
	TRAIN_DATA_PATH = "./data/train.csv"
	TEST_DATA_PATH = "./data/test.csv"

	def __init__(self):
		# Create a directed graph
		self.directed_graph = nx.read_edgelist(self.GRAPH_DATA_PATH, delimiter='\t', create_using=nx.DiGraph())
		self.train_ids = list()
		self.test_ids = list()
		self.y_train = list()
		self.n_train = 0
		self.n_test = 0

	def read_train_data(self):
		# Read training data
		with open(self.TRAIN_DATA_PATH, 'r') as f:
			next(f)
			for line in f:
				t = line.split(',')
				self.train_ids.append(t[0])
				self.y_train.append(t[1][:-1])
		
		self.n_train = len(self.train_ids)
		unique = np.unique(self.y_train)
		print("\nNumber of classes: ", unique.size)
	
	def read_test_data(self):
		# Read test data
		
		with open(self.TEST_DATA_PATH, 'r') as f:
			next(f)
			for line in f:
				self.test_ids.append(line[:-2])
		
		# Create the test matrix. Use the same 3 features as above
		self.n_test = len(self.test_ids)
	
	def create_features_matrix(self, mode="train"):
		"""Create the matrix. Each row corresponds to an article.
		Use the following 3 features for each article:
		(1) out-degree of node
		(2) in-degree of node
		(3) average degree of neighborhood of node
		"""
		
		if mode == "train":
			ids_list = self.train_ids
			number_of_articles = self.n_train
		else:
			ids_list = self.test_ids
			number_of_articles = self.n_test
		
		avg_neig_deg = nx.average_neighbor_degree(self.directed_graph, nodes=ids_list)
		X_data = np.zeros((number_of_articles, 3))
		for i in range(number_of_articles):
			X_data[i, 0] = self.directed_graph.out_degree(ids_list[i])
			X_data[i, 1] = self.directed_graph.in_degree(ids_list[i])
			X_data[i, 2] = avg_neig_deg[ids_list[i]]
			
		return X_data
	
	def create_graph_walk(self, mode="train"):
		walk_index = {}
		with open(os.path.join('./data/walk.embeddings')) as f:
			next(f)
			for line in f:
				values = line.split()
				node = values[0]
				coefs = np.asarray(values[1:], dtype='float32')
				walk_index[node] = coefs
		print('Found %s walk vectors.' % len(walk_index))
		
		if mode == "train" :
			ids_list = self.train_ids
			number_of_articles = self.n_train
		else:
			ids_list = self.test_ids
			number_of_articles = self.n_test
			
		train_graph_walk = np.zeros((number_of_articles, 64))
		for i in range(number_of_articles):
			train_graph_walk[i] = walk_index[ids_list[i]]

		return train_graph_walk


if __name__ == "__main__":
	
	graph_features = GraphFeatures()
	graph_features.read_train_data()
	graph_features.read_test_data()

	X_train = graph_features.create_features_matrix("train")
	X_test = graph_features.create_features_matrix("test")

	print("\nTrain matrix dimensionality: ", X_train.shape)
	print("Test matrix dimensionality: ", X_test.shape)
	X_train_df = pd.DataFrame(data=X_train, columns=['out_degre', 'in_degree', 'avg_neig_deg'])
	X_train_df['Article'] = graph_features.train_ids
