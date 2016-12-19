from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors
import numpy, sys, math
from pyspark.ml.feature import VectorAssembler

missing,total = 0,0

def lbfunc(line):
	global missing, total
	try:
		line = map(float, line.strip().split(','))
		label = line[4]
		line[6] = int(math.log(line[6],2))
		features = line[:4] + line[5:]
		total += 1
		return LabeledPoint(label, features)
	except:
		missing += 1
		total += 1
		# print line
		return None

def getRawData(sc):
	data = sc.textFile("data/new_processed_file.csv")
	header = data.first()
	data = data.filter(lambda row: row!=header).map(lambda line: lbfunc(line)).filter(lambda x:x!=None)
	print missing, total
	return data

def trainModel(sc, trainingData, testData):
	# Train a RandomForest model.
	#  Empty categoricalFeaturesInfo indicates all features are continuous.
	#  Note: Use larger numTrees in practice.
	#  Setting featureSubsetStrategy="auto" lets the algorithm choose.
	model = RandomForest.trainRegressor(trainingData, categoricalFeaturesInfo={5:7},
	                                    numTrees=500, featureSubsetStrategy="auto",
	                                    impurity='variance', maxDepth=6, maxBins=32)
	
	# Evaluate model on test instances and compute test error
	predictions = model.predict(testData.map(lambda x: x.features))
	labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
	
	testMSE = labelsAndPredictions.map(lambda (v, p): (v - p) * (v - p)).sum()/float(testData.count())
	print('Test Mean Squared Error = ' + str(testMSE))
	print('Learned regression forest model:')
	print(model.toDebugString())
	
	# Save and load model
	model.save(sc, "tmpModel/myRandomForestRegressionModel")
	return model

def plot(sc):
	model = RandomForestModel.load(sc, "tmpModel/myRandomForestRegressionModel")
	
	st,en = 0,6
	xarr = range(st,en+1)
	yarr = []
	sample = [8000,8000,6291.63,36,272.61,4,21000,0,11.43,0,2,7,0,5142,46.3,8,1,0,0,9088.73,6975.8,8000,1088.73,0,0,0,0,5547.71,5979]
	print sample
	from random import randint
	#loan = [8000,16000,24000,40000,48000]
	#term = [12,24,36,48,60]
	#grade = [0,1,2,3,4,5,6]
	income = [10000,20000,30000,40000,50000,60000,70000,80000,90000,100000]
	#for x in xarr:
		#temp = sample[:]
		#temp[5] = x
		#temp[0] *= randint(1,5)
		#print temp[0]
		#yarr.append(model.predict(temp))
	
	for x in income:
		temp = sample[:]
		temp[6] = x
		yarr.append(model.predict(temp))
	# Generate Graph from xarr and yarr
	#print xarr,yarr
	print income,yarr

def main():
	sc = SparkContext(appName="PythonRandomForestRegressionExample")
	data = getRawData(sc)
	
	# Split the data into training and test sets (30% held out for testing)
	(trainingData, testData) = data.randomSplit([0.7, 0.3])
	model = trainModel(sc, trainingData, testData)
	plot(sc)

if __name__ == '__main__':
	print 'yoy'
	main()
