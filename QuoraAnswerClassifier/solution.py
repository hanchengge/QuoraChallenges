# author: Hancheng Ge
# March 28th, 2017

import fileinput
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score
from sklearn.cross_validation import train_test_split

class Solution(object):
	def __init__(self):
		self.numTrain = None
		self.numFeatures = None
		self.numTest = None
		self.trainX = []
		self.trainY = []
		self.trainID = []
		self.testX = []
		self.testID = []
		self.testY = []
		self.models = []
		self.modelNames = []
		self.bestModel = None


	def readData(self, filePath, online):
		# read the data from the training and testing data
		lineNum = 0
		flag_test = 0
		if not online:
			f = open(filePath)
		else:
			f = fileinput.input()
		for line in f:
			try:
				if lineNum == 0:
					# extract the number of training samples and the number of features
					if not flag_test:
						self.numTrain, self.numFeatures = [int(x) for x in line.strip().split()]
					else:
						self.numTest = int(line.strip())
				else:
					# if not for the testing data, read each line for the training data
					if not flag_test:
						raw = line.strip().split()
						userID = raw[0]
						label = int(raw[1])
						featureVec = [0]*self.numFeatures
						for feature in raw[2:]:
							raw_feature = feature.split(":")
							featureIdx = int(raw_feature[0])
							featureVal = float(raw_feature[1])
							featureVec[featureIdx-1] = featureVal
						self.trainX.append(featureVec)
						self.trainY.append(label)
						self.trainID.append(userID)
						if lineNum == self.numTrain:
							lineNum = -1
							flag_test = 1
					# if it is for the testing data, read each line for the testing data
					else:
						raw = line.strip().split()
						userID = raw[0]
						featureVec = [0]*self.numFeatures
						for feature in raw[1:]:
							raw_feature = feature.split(":")
							featureIdx = int(raw_feature[0])
							featureVal = float(raw_feature[1])
							featureVec[featureIdx-1] = featureVal
						self.testX.append(featureVec)
						self.testID.append(userID)
				lineNum += 1
			except Exception as err:
				print str(err)
		if filePath:
			f.close()
		# check if it is a balanced dataset
		numPositiveSamples = sum([1 for _ in self.trainY if _ == 1])
		numNegativeSamples = self.numTrain - numPositiveSamples
		# print "STATUS: all data has been read into the memory with {} samples of +1 and {} samples of -1.".format(numPositiveSamples, numNegativeSamples)
		if numPositiveSamples == numNegativeSamples:
			# print "STATUS: we have a balanced data."
			pass
		else:
			# print "STATUS: we have an unbalanced data; an oversampling technique should be applied."
			pass
		self.trainX = np.array(self.trainX)[:,:-2]
		self.trainY = np.array(self.trainY)
		self.trainID = np.array(self.trainID)
		self.testX = np.array(self.testX)[:,:-2]
		self.testID = np.array(self.testID)


	def preProcess(self):
		# Standardize
		X = np.concatenate((self.trainX, self.testX), axis=0)
		meanVal = X.mean(axis=0)
		stdVal = X.std(axis=0)
		X = (X - meanVal) / stdVal
		self.trainX = X[:self.numTrain,:]
		self.testX = X[self.numTrain:,:]


	def featureSelection(self):
		# utilize LASSO to perform the feature selection
		clf = LassoCV()
		sfm = SelectFromModel(clf, threshold=0.1)
		sfm.fit(self.trainX, self.trainY)
		n_features = sfm.transform(self.trainX).shape[1]
		while n_features > 10:
			sfm.threshold += 0.1
			X_transform = sfm.transform(self.trainX)
			n_features = X_transform.shape[1]
		self.trainX = sfm.transform(self.trainX)
		self.testX = sfm.transform(self.testX)
		# print "Status: feature selection has completed with selected {} features!".format(n_features)
		# print "Status: select features such as {}.".format(np.array(range(self.numFeatures-2))[sfm.get_support()])


	def cross_val_score(self, trainX, trainY, cv, scoring):
		# cross validation to select the best model
		scores = [[] for _ in range(len(self.models))]
		for _ in range(cv):
			X_train, X_test, Y_train, Y_test = train_test_split(trainX, trainY, test_size=.4, random_state=42)
			idx = 0
			for name, clf in zip(self.modelNames, self.models):
				clf.fit(X_train, Y_train)
				scores[idx].append(clf.score(X_test, Y_test))
				idx += 1
		maxScore = 0
		for i, score in enumerate(scores):
			avgScore = np.mean(score)
			# print "STATUS: F1 measure for the model {} in the training is {}.".format(self.modelNames[i], avgScore)
			if avgScore > maxScore:
				maxScore = avgScore
				self.bestModel = (self.modelNames[i], self.models[i])
		# print "STATUS: the the model {} performs the best in the training.".format(self.bestModel[0])


	def modelTraining(self, modelSelect=None):
		# model training and model selection by comparing different models with the cross-validation
		n_estimators = 100
		kfold = 5
		self.modelNames = ["Linear SVM", "RBF SVM", "Decision Tree", "Random Forest", "Extra Tree", "AdaBoost","Naive Bayes"]
		self.models = [
						SVC(kernel="linear", C=0.025),\
						SVC(gamma=2, C=1),\
						DecisionTreeClassifier(max_depth=None),\
						RandomForestClassifier(n_estimators=n_estimators),\
						ExtraTreesClassifier(n_estimators=n_estimators),\
						AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=n_estimators),\
						GaussianNB()\
					  ]
		try:
			if not modelSelect:
				self.cross_val_score(self.trainX, self.trainY, cv=kfold, scoring='f1')
			else:
				idx = self.modelNames.index(modelSelect)
				self.bestModel = (modelSelect, self.models[idx])
				# print "STATUS: the model {} is selected.".format(modelSelect)
			self.bestModel[1].fit(self.trainX, self.trainY)
			# print "STATUS: the model {} has been successfully trained.".format(self.bestModel[0])
		except Exception as err:
			print str(err)


	def readOutput(self, filePath):
		# read the ground truth of the testing data
		self.testY = [0]*self.numTest
		with open(filePath) as f:
			for line in f:
				raw = line.strip().split()
				userID = raw[0]
				groundTruth = int(raw[1])
				idx = np.where(self.testID==userID)[0][0]
				self.testY[idx] = groundTruth
		# print "STATUS: the grand truth of testing data has been imported into the memory."


	def prediction(self, online=False):
		try:
			# predict if a user will answer a question by using the selected model
			predictions = self.bestModel[1].predict(self.testX)
			# calculate the f1 meausre to evaluate the performance of the selected model in the testing data
			if not online:
				avgF1 = f1_score(self.testY, predictions, average='weighted')
				# print "STATUS: F1 measure for the testing data is {}.".format(avgF1)
			# print "STATUS: predicted Answers are as follows:"
			for userID, predict in zip(self.testID, predictions):
				print userID + " " + "+1" if predict==1 else userID + " " + "-1"
		except Exception as err:
			print str(err)


if __name__ == '__main__':
	trainDataPath = "input00.txt"
	testOutputPath = "output00.txt"
	online = True
	sol = Solution()
	sol.readData(trainDataPath, online=online)
	sol.preProcess()
	# sol.featureSelection()
	sol.modelTraining("Random Forest")
	# sol.modelTraining()
	if not online:
		sol.readOutput(testOutputPath)
	sol.prediction(online=online)