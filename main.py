import sys
import math
import numpy

from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SQLContext
from pyspark import SparkContext

from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel


def lbfunc(line):
    global missing, total
    try:
        line = map(float, line.strip().split(','))
        label = line[4]
        line[6] = int(math.log(line[6], 2))
        features = line[:4] + line[5:]
        return LabeledPoint(label, features)
    except:
        return None


def writeS3(LOGGER, msg):
    LOGGER.info('StdOut: {}'.format(msg))


def getRawData(sc, repeat=3):
    data = sc.textFile("s3n://bigdatap2ploans/new_processed_file.csv")
    # data = sc.textFile("data/augmented_file.csv")
    # data = sc.textFile("data/new_processed_file.csv")
    header = data.first()
    data = data.filter(lambda row: row != header).map(lambda line: lbfunc(line)).filter(lambda x: x != None)
    final_data = data
    
    for k in xrange(repeat):
        final_data = final_data.union(data)
    return final_data


def trainModelRF(sc, trainingData, testData, LOGGER, numT=50):
    # Train a RandomForest model.
    #  Empty categoricalFeaturesInfo indicates all features are continuous.
    #  Note: Use larger numTrees in practice.
    #  Setting featureSubsetStrategy="auto" lets the algorithm choose.
    model = RandomForest.trainRegressor(trainingData, categoricalFeaturesInfo={5: 7},
                                        numTrees=numT, featureSubsetStrategy="auto",
                                        impurity='variance', maxDepth=20, maxBins=32)

    # Evaluate model on test instances and compute test error
    predictions = model.predict(testData.map(lambda x: x.features))
    labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)

    testMSE = labelsAndPredictions.map(lambda (v, p): (v - p) * (v - p)).sum() / float(testData.count())
    writeS3(LOGGER, 'Test Mean Squared Error = ' + str(testMSE))
    writeS3(LOGGER, 'Learned regression forest model:')
    writeS3(LOGGER, model.toDebugString())

    # Save and load model
    model.save(sc, "s3n://bigdatap2ploans/rf")
    # model.save(sc, "model/rf")
    return model


def trainModelGBT(sc, trainingData, testData, LOGGER, itera=50):
    # Train a GradientBoostedTrees model.
    #  Notes: (a) Empty categoricalFeaturesInfo indicates all features are continuous.
    #         (b) Use more iterations in practice.
    model = GradientBoostedTrees.trainRegressor(trainingData, categoricalFeaturesInfo={5: 7}, numIterations=itera)

    # Evaluate model on test instances and compute test error
    predictions = model.predict(testData.map(lambda x: x.features))
    labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
    testMSE = labelsAndPredictions.map(lambda (v, p): (v - p) * (v - p)).sum() / float(testData.count())
    writeS3(LOGGER, 'Test Mean Squared Error = ' + str(testMSE))
    writeS3(LOGGER, 'Learned regression GBT model:')
    writeS3(LOGGER, model.toDebugString())

    # Save and load model
    model.save(sc, "s3n://bigdatap2ploans/gbt")
    # model.save(sc, "model/gbt")
    return model


def plot(sc, model, LOGGER):
    # modelRF = RandomForestModel.load(sc, "s3n://bigdatap2ploans/myRandomForestRegressionModel")

    st, en = 0, 6
    xarr = range(st, en + 1)
    yarr = []
    sample = [8000, 8000, 6291.63, 36, 272.61, 4, 21000, 0, 11.43, 0, 2, 7, 0, 5142,
              46.3, 8, 1, 0, 0, 9088.73, 6975.8, 8000, 1088.73, 0, 0, 0, 0, 5547.71, 5979]

    writeS3(LOGGER, sample)
    from random import randint
    # loan = [8000,16000,24000,40000,48000]
    # term = [12,24,36,48,60]
    # grade = [0,1,2,3,4,5,6]
    income = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
    # for x in xarr:
    # temp = sample[:]
    # temp[5] = x
    # temp[0] *= randint(1,5)
    # print temp[0]
    # yarr.append(model.predict(temp))

    for x in income:
        temp = sample[:]
        temp[6] = x
        yarr.append(model.predict(temp))
    # Generate Graph from xarr and yarr
    # print xarr,yarr
    writeS3(LOGGER, income)
    writeS3(LOGGER, yarr)


if __name__ == '__main__':

    sc = SparkContext(appName="BigData")
    log4jLogger = sc._jvm.org.apache.log4j
    LOGGER = log4jLogger.LogManager.getLogger(__name__)

    writeS3(LOGGER, 'Fetching Data')
    data = getRawData(sc)

    writeS3(LOGGER, 'Split the data into training and test sets (.30 held out for testing)')
    (trainingData, testData) = data.randomSplit([0.7, 0.3])

    writeS3(LOGGER, 'training RF')
    modelRF = trainModelRF(sc, trainingData, testData, LOGGER)

    writeS3(LOGGER, 'training GBT')
    modelGBT = trainModelGBT(sc, trainingData, testData, LOGGER)

    writeS3(LOGGER, 'Plot')
    plot(sc, modelRF, LOGGER)
    plot(sc, modelGBT, LOGGER)
    writeS3(LOGGER, 'Done')
