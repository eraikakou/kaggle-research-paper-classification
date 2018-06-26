import pandas as pd
from graph_features import GraphFeatures


class DataLoader(object):

	DATA_NODE_PATH = "./data/node_information.csv"
	TRAIN_DATA_PATH = "./data/train.csv"
	TEST_DATA_PATH = "./data/test.csv"

	def __init__(self):
		self.data_node = pd.read_csv(self.DATA_NODE_PATH)
		self.train_data = pd.read_csv(self.TRAIN_DATA_PATH)
		self.test_data = pd.read_csv(self.TEST_DATA_PATH)
		
		graph_train_dataframe , graph_test_dataframe = self.collect_train_and_test_data_graph_features()
		
		# Merge the dataframe that includes the labels with the dataframe that includes
		# the graph features for train and test dataset
		self.completed_train_data = pd.merge(self.train_data, graph_train_dataframe, how='left', left_on=['Article'], right_on=['Article'])
		self.completed_test_data = pd.merge(self.test_data, graph_test_dataframe, how='left', left_on=['Article'], right_on=['Article'])
		
		# Merge the dataframe that includes the labels and the graph features with the dataframe that includes the data node
		# information (e.g. title, abstract, authors) for train and test dataset
		self.train_data_labels_info = pd.merge(self.completed_train_data, self.data_node, how='left', left_on=['Article'], right_on=['id'])
		self.test_data_labels_info = pd.merge(self.completed_test_data, self.data_node, how='left', left_on=['Article'], right_on=['id'])
	
	def split_train_dataset(self, train_percentage=0.8, test_percentage=0.2):
		"""Split the train dataset into train and test dataset
		The default percentages are: train - 80% and test - 20%
		"""
	
		if train_percentage + test_percentage <= 1:
			self.train_data_labels_info = self.train_data_labels_info.sample(frac=1, random_state=1).reset_index(drop=True)
			train_papers = self.train_data_labels_info[:int(train_percentage * len(self.train_data_labels_info))]
			test_papers = self.train_data_labels_info[int(train_percentage * len(self.train_data_labels_info)):]
		
			return train_papers, test_papers

		else:
			return []

	@staticmethod
	def collect_train_and_test_data_graph_features():
		"""Collect some features from the given graph for train
		and test dataset.
		"""
		
		graph_features = GraphFeatures()
		graph_features.read_train_data()
		graph_features.read_test_data()
		
		X_train = graph_features.create_features_matrix("train")
		X_test = graph_features.create_features_matrix("test")
		
		print("\nTrain matrix dimensionality: ", X_train.shape)
		print("Test matrix dimensionality: ", X_test.shape)
		X_train_df = pd.DataFrame(data=X_train, columns=['out_degree', 'in_degree', 'avg_neig_deg'])
		X_train_df['Article'] = graph_features.train_ids
		X_train_df['Article'] = X_train_df['Article'].astype('int64')
		
		X_test_df = pd.DataFrame(data=X_test, columns=['out_degree', 'in_degree', 'avg_neig_deg'])
		X_test_df['Article'] = graph_features.test_ids
		X_test_df['Article'] = X_test_df['Article'].astype('int64')
		
		return X_train_df, X_test_df


if __name__== "__main__":
	
	dataLoader = DataLoader()
	train_papers = None
	test_papers = None
	split_data = dataLoader.split_train_dataset()
	
	if len(split_data) > 0:
		train_papers = split_data[0]
		test_papers = split_data[1]
