


import sys

import sklearn
#sys.path.append('new_codonSlider.py')
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.svm import NuSVC
#from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
# command line (file, DNAseqFile, window length)

aFile = open(sys.argv[1],'r')
outfile = open(sys.argv[2],'w')
startWindowLocation = int(sys.argv[3])
stopWindowLocation = int(sys.argv[4])

dnastr = ""
linecounter = 0
for line in aFile:
	if line[0] != '>' and linecounter == 0:
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
	feature_vals = [[0 for x in range(window)]  for x in range(len(dnastr)-window)]
	labeled_data = []
	line_counter = 0
	
	while lengthVar < len(dnastr):
		anotherstr = dnastr[counter:counter+window]	
		nuc_counter = 0
		while nuc_counter < window:
			if nuc_counter == window -1:
				feature_vals[counter][nuc_counter] = float(anotherstr[nuc_counter])
			else:
				feature_vals[counter][nuc_counter] = float(anotherstr[nuc_counter])
			nuc_counter += 1
		if counter < len(anotherstr)-1:
			labeled_data.append(float(dnastr[counter+window+1]))
		else:
			labeled_data.append(float(dnastr[counter+window]))
		counter += 1
		lengthVar = counter + window
	
	return feature_vals, labeled_data


outfile.write("Window size\tDT Class\tRF Class\tExT Class\tSVM Class\tBayes Class\n")

for x in range(startWindowLocation,stopWindowLocation):
	print str(x) + " iteration"
	something = pipe(x)
	
	ohe_features = OneHotEncoder()
	ohe_labels = OneHotEncoder()
	ohe_features = ohe_features.fit_transform(something[0]).toarray()
	
	X = ohe_features # all features 
	y = something[1] # all labels

	# if test_size/train_size are null then test_size == .75
	#X_train, X_test, y_train, y_test = train_test_split(X,y)

	dtClf = DecisionTreeClassifier(random_state=0)
	dtScore = cross_val_score(dtClf,X,y).mean()	

	rfClf = RandomForestClassifier(n_estimators=100,random_state=0)
	rfScores = cross_val_score(rfClf,X,y).mean()

	etClf = ExtraTreesClassifier(random_state=0)
	etScores = cross_val_score(etClf,X,y).mean()
	
	svmClf = SVC()
	svmScore = cross_val_score(svmClf,X,y).mean()
	
	naiveBClf = GaussianNB()
	nbScore = cross_val_score(naiveBClf,X,y).mean()

	outfile.write(str(x) +"\t" + str(dtScore) + "\t" + str(rfScores) + "\t" + str(etScores) + "\t" + str(svmScore) +"\t" + str(nbScore) +"\n")

outfile.close()



