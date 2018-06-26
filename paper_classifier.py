from custom_transformers import *
from data_loader import DataLoader
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import make_scorer, log_loss, roc_curve, auc, roc_auc_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
import csv


class PaperClassifier(object):
	
	TRAIN_DATA_PATH = "./data/train.csv"
	
	def __init__(self):
		dataLoader = DataLoader()
		train_papers = None
		test_papers = None
		split_data = dataLoader.split_train_dataset()
		
		if len(split_data) > 0:
			train_papers = split_data[0]
			test_papers = split_data[1]
		
		self.x_train = self.apply_feature_extraction(train_papers)
		self.y_train = train_papers["Journal"].tolist()
		
		self.x_train_test = self.apply_feature_extraction(test_papers)
		self.y_train_test = test_papers["Journal"].tolist()
		
		self.x_test = self.apply_feature_extraction(dataLoader.test_data_labels_info)
		
		# self.text_clf = LogisticRegression()
		# self.parameters_grid = {
		# 	'classifier__C': [0.01, 0.1, 1.0],
		# 	'classifier__class_weight': [None, 'balanced'],
		# 	'classifier__penalty': ['l1', 'l2'],
		# 	'classifier__tol': [0.001, 0.0001],
		# }
		
		self.text_clf = LogisticRegression(penalty='l2')
		self.parameters_grid = {
			'classifier__tol': [0.001, 0.0001],
		}
		
		self.thres_all = None
		self.logr_model = LogisticRegression(penalty='l2', tol=1e-05)
		self.pipeline = Pipeline([
			# Use FeatureUnion to combine the features from subject and body
			('union', FeatureUnion(
				transformer_list=[
					('number_of_authors', Pipeline([
						('selector', NumberSelector(field='num_auth')),
						('normalizer', StandardScaler(copy=True, with_mean=True, with_std=True)),
					])),
					# Pipeline for pulling features from the paper's title
					('title', Pipeline([
						('selector', TextSelector(field='title')),
						('preprocessor', TextPreprocessor()),
						('vectorizer', TfidfVectorizer(
							tokenizer=self.identity, preprocessor=None, lowercase=False, max_df=0.6, min_df=0.001,
							ngram_range=(1, 2)
						)),
						('sfm_abs', SelectFromModel(self.logr_model, threshold=self.thres_all))
					])),
					# Pipeline for pulling features from authors
					('author', Pipeline([
						('selector', TextSelector(field='authors')),
						('vect', CountVectorizer(decode_error='ignore', stop_words='english', max_df=0.03, min_df=0, ngram_range=(1, 2))),
						('tfidf', TfidfTransformer(norm='l2', sublinear_tf=True)),
						('sfm_aut', SelectFromModel(self.logr_model, threshold=0.55)),
					])),
					('avg_word_len_auth', Pipeline([
						('selector', NumberSelector(field='avg_word_len_auth')),
						('normalizer', StandardScaler(copy=True, with_mean=True, with_std=True)),
					])),
					('out_degree', Pipeline([
						('selector', NumberSelector(field='out_degree')),
						('normalizer', StandardScaler(copy=True, with_mean=True, with_std=True)),
					])),
					('in_degree', Pipeline([
						('selector', NumberSelector(field='in_degree')),
						('normalizer', StandardScaler(copy=True, with_mean=True, with_std=True)),
					])),
					('avg_neig_deg', Pipeline([
						('selector', NumberSelector(field='avg_neig_deg')),
						('normalizer', StandardScaler(copy=True, with_mean=True, with_std=True)),
					])),
					('abstract', Pipeline([
						('selector', TextSelector(field='abstract')),
						('preprocessor', TextPreprocessor()),
						('vectorizer', TfidfVectorizer(tokenizer=self.identity, preprocessor=None, lowercase=False, max_df=0.6, min_df=0.001, ngram_range=(1, 2))),
						('sfm_abs', SelectFromModel(self.logr_model, threshold=self.thres_all))
					])),
				],
				
				# weight components in FeatureUnion
				transformer_weights={
					'title': 1.65,
					'out_degree': 0.7,
					'in_degree': 0.9,
					'avg_neig_deg': 0.8,
					'abstract': 1.65,
					'author': 1.5
				},
			)),
			
			# Use a LogisticRegression classifier on the combined features
			('classifier', self.text_clf),
		])
	
	def apply_feature_extraction(self, df):
		df = df.replace(np.nan, '', regex=True)
		df["avg_word_len_auth"] = df["authors"].apply(self.average_word_length)
		df["num_auth"] = df["authors"].apply(self.number_of_authors)
		df = df.fillna(0)
		
		return df
	
	@staticmethod
	def identity(arg):
		"""
		Simple identity function works as a passthrough.
		"""
		return arg
	
	@staticmethod
	def average_word_length(name):
		"""Helper code to compute average word length of a name"""
		if len(name) == 0:
			return 0
		return np.mean([len(word) for word in name.split()])
	
	@staticmethod
	def number_of_authors(names):
		num_of_authors = 0
		
		if len(names) > 0:
			num_of_authors = names.count(",") + 1
		
		return num_of_authors

	def get_unique_labels(self):
		# Read training data
		train_ids = list()
		y_train = list()
		with open(self.TRAIN_DATA_PATH, 'r') as f:
			next(f)
			for line in f:
				t = line.split(',')
				train_ids.append(t[0])
				y_train.append(t[1][:-1])
				
		return np.unique(y_train)
	
	def plot_roc_curves(self, y_train_test_score):
		lw = 2
		
		# Binarize the output - labels
		unique_labels = self.get_unique_labels().tolist()
		y_train_test = label_binarize(self.y_train_test, classes=unique_labels)
		n_classes_train_test = y_train_test.shape[1]
		
		# Compute ROC curve and ROC area for each class
		fpr = dict()
		tpr = dict()
		roc_auc = dict()
		for i in range(n_classes_train_test):
			fpr[i], tpr[i], _ = roc_curve(y_train_test[:, i], y_train_test_score[:, i])
			roc_auc[i] = auc(fpr[i], tpr[i])
		
		# Compute micro-average ROC curve and ROC area
		fpr["micro"], tpr["micro"], _ = roc_curve(y_train_test.ravel(), y_train_test_score.ravel())
		roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
		
		# First aggregate all false positive rates
		all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes_train_test)]))
		
		# Then interpolate all ROC curves at this points
		mean_tpr = np.zeros_like(all_fpr)
		for i in range(n_classes_train_test):
			mean_tpr += interp(all_fpr, fpr[i], tpr[i])
		
		# Finally average it and compute AUC
		mean_tpr /= n_classes_train_test
		
		fpr["macro"] = all_fpr
		tpr["macro"] = mean_tpr
		roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
		
		for i in range(n_classes_train_test):
			plt.figure()
			lw = 2
			plt.plot(fpr[i], tpr[i], color='darkorange',
			         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[i])
			plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
			plt.xlim([0.0, 1.0])
			plt.ylim([0.0, 1.05])
			plt.xlabel('False Positive Rate')
			plt.ylabel('True Positive Rate')
			plt.title('Receiver operating characteristic for: ' + unique_labels[i])
			plt.legend(loc="lower right")
			plt.show()


if __name__== "__main__":
	
	paperClassifier = PaperClassifier()
	
	# Imbalanced Data
	fig = plt.figure(figsize=(8, 6))
	paperClassifier.x_train.groupby('Journal').abstract.count().plot.bar(ylim=0)
	plt.show()
	
	# Define the as scorer function the log loss. It'll be used to cross validation
	log_loss_scorer = make_scorer(log_loss, greater_is_better=False, needs_proba=True)

	# Cross Validation through Grid Search - tune the Hyper-Parameters
	grid_search = GridSearchCV(paperClassifier.pipeline, param_grid=paperClassifier.parameters_grid, n_jobs=-1, verbose=10, scoring=log_loss_scorer)
	grid_search = grid_search.fit(paperClassifier.x_train, paperClassifier.y_train)

	# Find the best Hyper Parametets for the estimator
	print("Best score: %0.3f" % grid_search.best_score_)
	print("Best parameters set:")
	best_parameters = grid_search.best_estimator_.get_params()
	for param_name in sorted(paperClassifier.parameters_grid.keys()):
		print("\t%s: %r" % (param_name, best_parameters[param_name]))

	paperClassifier.text_clf = LogisticRegression(penalty='l2', tol=best_parameters['classifier__tol'])

	# Fit the model OneVsRestClassifier
	paper_clf = OneVsRestClassifier(paperClassifier.pipeline).fit(paperClassifier.x_train, paperClassifier.y_train)
	y_train_test_score = paper_clf.decision_function(paperClassifier.x_train_test)

	paperClassifier.plot_roc_curves(y_train_test_score)

	# Get test IDs too
	test_ids = list()
	with open('./data/test.csv', 'r') as f:
		next(f)
		for line in f:
			test_ids.append(line[:-2])

	y_pred = paper_clf.predict_proba(paperClassifier.x_test)

	# Write predictions to a file
	with open('sample_submission.csv', 'w') as csvfile:
		writer = csv.writer(csvfile, delimiter=',')
		lst = paper_clf.classes_.tolist()
		lst.insert(0, "Article")
		writer.writerow(lst)
		for i, test_id in enumerate(test_ids):
			lst = y_pred[i, :].tolist()
			lst.insert(0, test_id)
			writer.writerow(lst)
