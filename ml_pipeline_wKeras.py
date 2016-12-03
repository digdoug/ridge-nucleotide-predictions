#!/usr/bin/env python

import sys
import sklearn
import keras

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential 
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

# command line (file, DNAseqFile, window length)

if len(sys.argv) < 5:
	print "USAGE: DNAseq.txt, predictionResults.txt, startwindow, stopwindow"
	sys.exit()

aFile = open(sys.argv[1],'r')
outfile = open(sys.argv[2],'w')
startWindowLocation = int(sys.argv[3])
stopWindowLocation = int(sys.argv[4])

dnastr = ""
linecounter = 0
for line in aFile:
	if line[0] != '>'and linecounter == 0:
		line = line.strip()	
		dnastr += line
		linecounter = linecounter + 1
		if linecounter == 10:
			linecounter = 0

aFile.close()
dnastr = list(dnastr)
# label encoding
le = preprocessing.LabelEncoder()
le.fit(dnastr)
dnastr = ''.join(str(v) for v in le.transform(dnastr))
		
def pipe(window):

	counter = 0
	lengthVar = 0
	feature_vals = [[0 for x in range(window)]  for x in range(((len(dnastr)-window)/10))]
	labeled_data = []
	line_counter = 0
	downSampler = 0

	
	while lengthVar < len(dnastr):
		anotherstr = dnastr[counter:counter+window]	
		nuc_counter = 0
		if counter % 10 == 0 and downSampler < counter /10:
			while nuc_counter < window :
				feature_vals[downSampler][nuc_counter] = float(anotherstr[nuc_counter])
				nuc_counter += 1
			if counter < len(anotherstr)-1:
				labeled_data.append(float(dnastr[downSampler+window+1]))
			else:
				labeled_data.append(float(dnastr[downSampler+window]))
			downSampler += 1	
		counter += 1
		lengthVar = counter + window
		
	if len(feature_vals) != len(labeled_data):
		del feature_vals[-1]
	return feature_vals, labeled_data


outfile.write("Window size\tDT Class\tRF Class\tExT Class\tSVM Class\tBayes Class\tMLP Class\n")

for x in range(startWindowLocation,stopWindowLocation):
	print str(x) + " iteration"
	something = pipe(x)
	
	ohe_features = OneHotEncoder()
	ohe_features = ohe_features.fit_transform(something[0]).toarray()
	
	X = ohe_features # all features 
	y = something[1] # all labels

	# if test_size/train_size are null then tain_size == .75
	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.3)

	
	# recurrent neural net
	model = Sequential() 
	# Dense(64) is a fully-connected layer with 64 hidden units.
	# in the first layer, you must specify the expected input data shape:
	# here, 20-dimensional vectors.
	model.add(Dense(output_dim=64, input_dim=100, init="glorot_uniform"))
	model.add(Activation("relu"))
	model.add(Dense(output_dim=10, init="glorot_uniform"))
	model.add(Activation("softmax"))
	
	model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))

	print X_train.shape
	print len(y_train)
	X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=None, dtype='int32')
	model.fit(X_train, y_train,
			          nb_epoch=5,
					            batch_size=32)
	RNNscore = model.evaluate(X_test, y_test, batch_size=32)



#	outfile.write(str(x) +"\t" + str(dtScore) + "\t" + str(rfScores) + "\t" + str(etScores) + "\t" + str(svmScore) +"\t" + str(nbScore) +"\t"+str(mlpScore)+"\t"+str(RNNscore)+"\n")





