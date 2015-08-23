import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.datasets import load_files
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
import csv
from sklearn import cross_validation
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.feature_extraction.text import CountVectorizer	
if __name__ == "__main__":
	# Initialize 	
	data = []
	target = []
	sent_sum = 0
	final_test = []
	final_target = []
	#Cval = float(sys.argv[3]) # ideal value is 0.55
	#Read input file and break into train data and labels
	with open(sys.argv[1]) as csvfile:
		linereader = csv.reader(csvfile, delimiter='\t')
		sentence_len = 0
		for row in linereader:
			data.append(row[2])
			target.append(row[3])
			sent_sum += len(row[2])
	avg_len = sent_sum/len(data)
	#Read test file and break into test_predict and test_labels
	with open(sys.argv[2]) as csvfile:
		linereader = csv.reader(csvfile, delimiter='\t')
		for row in linereader:
			final_test.append(row[2])
			final_target.append(row[3])
        # split the data to form a base for crossvalidation to be done in gridsearchCV
	train_data, test_data, train_label, test_label = train_test_split(
		data, target, test_size=0.25, random_state=42)

        #create a pipeline for all the tasts to be performed - calculate TFIDF and form bag of words - select classifier and parameters
	pipeline_AllProcesses = Pipeline([
		('vect', TfidfVectorizer(min_df=3, max_df=0.95)),
		('clf', OneVsOneClassifier(LinearSVC(penalty='l2', dual=True, C=0.55, fit_intercept=True, intercept_scaling=1, 	class_weight=None, verbose=0, random_state=None))),
		])

        #parameter tuning
	parameters_grid = {
		'vect__ngram_range': [(1, 1), (1, 2)],
	}

	#create crossvalidation sets
	grid_search = GridSearchCV(pipeline_AllProcesses, parameters_grid, n_jobs=-1,cv=10)

	#SVM Learner
	grid_search.fit(train_data, train_label)
	#prints model parameters for all parameters the model is being tuned with
	print(grid_search.grid_scores_)

        #predict the validation set
	test_predicted = grid_search.predict(test_data)

        #print Accuracy
	print(accuracy_score(test_label, test_predicted))

	unique_labels = ['neg','little-neg','neutral','little-pos','pos']
	
    	#print f1-measure, precision and recall
	print(metrics.classification_report(test_label, test_predicted,
	target_names=unique_labels))

	#calculate and print confusion matrix for training data
	conf_Mat = metrics.confusion_matrix(test_label, test_predicted)
	print(conf_Mat)

	#predict the output for final test set using SVM Classifier
	final_predicted = grid_search.predict(final_test)
	print(accuracy_score(final_target, final_predicted))

	print(metrics.classification_report(final_target, final_predicted,
	target_names=unique_labels))
