import sys
import math
import numpy
import argparse

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
    print msg


def getRawData(sc, file_name="s3n://bigdatap2ploans/new_processed_file_2.csv"):
    data = sc.textFile(file_name).cache()
    # data = sc.textFile("data/augmented_file.csv")
    # data = sc.textFile("data/new_processed_file.csv")
    header = data.first()
    data = data.filter(lambda row: row != header).map(lambda line: lbfunc(line)).filter(lambda x: x != None)
    return data


def trainModelRF(sc, trainingData, testData, LOGGER, numT=500):
    # Train a RandomForest model.
    #  Empty categoricalFeaturesInfo indicates all features are continuous.
    #  Note: Use larger numTrees in practice.
    #  Setting featureSubsetStrategy="auto" lets the algorithm choose.
    model = RandomForest.trainRegressor(trainingData, categoricalFeaturesInfo={5: 7},
                                        numTrees=numT, featureSubsetStrategy="auto",
                                        impurity='variance', maxDepth=10, maxBins=32)

    # Evaluate model on test instances and compute test error
    predictions = model.predict(testData.map(lambda x: x.features))
    labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)

    testMSE = labelsAndPredictions.map(lambda (v, p): (v - p) * (v - p)).sum() / float(testData.count())
    writeS3(LOGGER, 'Test Mean Squared Error = ' + str(testMSE))
    writeS3(LOGGER, 'Learned regression forest model:')
    writeS3(LOGGER, model.toDebugString())

    # Save and load model
    model.save(sc, "s3n://bigdatap2ploans/model_3/rf")
    # model.save(sc, "model/rf")
    return model


def trainModelGBT(sc, trainingData, testData, LOGGER, itera=100):
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
    model.save(sc, "s3n://bigdatap2ploans/model_3/gbt")
    # model.save(sc, "model/gbt")
    return model


def plot(sc, model, LOGGER):
    st, en = 0, 6

    sample = [8000,8000,6291.63,36,272.61,4,21000,0,11.43,0,2,7,0,5142,46.3,8,1,0,0,9088.73,6975.8,8000,1088.73,0,0,0,0,5547.71,5979]

    writeS3(LOGGER, sample)
    loan = [8000, 16000, 24000, 40000, 48000, 100000]
    term = [6,12, 24, 36, 48, 60]
    grade = range(7)
    income = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]

    yarr = []
    for x in loan:
        temp = sample[:]
        temp[0] = x
        temp[1] = x
        yarr.append(model.predict(temp))

    writeS3(LOGGER, 'Loan')
    writeS3(LOGGER, loan)
    writeS3(LOGGER, yarr)

    yarr = []
    for x in income:
        temp = sample[:]
        temp[6] = x
        yarr.append(model.predict(temp))

    writeS3(LOGGER, 'Income')
    writeS3(LOGGER, income)
    writeS3(LOGGER, yarr)

    yarr = []
    for x in grade:
        temp = sample[:]
        temp[5] = x
        yarr.append(model.predict(temp))

    writeS3(LOGGER, 'Grade')
    writeS3(LOGGER, grade)
    writeS3(LOGGER, yarr)

    yarr = []
    for x in term:
        temp = sample[:]
        temp[3] = x
        yarr.append(model.predict(temp))

    writeS3(LOGGER, 'Term')
    writeS3(LOGGER, term)
    writeS3(LOGGER, yarr)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Big Data')
    parser.add_argument('--train', required=True, metavar='t', type=int, nargs=1, help='Train on provided file, must be a csv with header; 1: RF, 2:GBT; 3:Both')
    parser.add_argument('--plot', required=True, metavar='p', type=int, nargs=1, help='Generate prediction Data')
    parser.add_argument('--file', metavar='f', type=str, nargs=1, help='Dataset File; can be S3, HDFS or Local File;Default: Sample file in S3')

    parser.add_argument('--numT',metavar='rft', type=int, nargs=1, help='Number of trees in RF; default=500')
    parser.add_argument('--iter',metavar='gbi', type=int, nargs=1, help='Number of iterations in GBT; default=100')

    args = parser.parse_args()

    sc = SparkContext(appName="BigData")
    log4jLogger = sc._jvm.org.apache.log4j
    LOGGER = log4jLogger.LogManager.getLogger(__name__)

    if args.train[0]:
        writeS3(LOGGER, 'Fetching Data')

        if args.file:
            data = getRawData(sc, args.file[0])
        else:
            data = getRawData(sc)

        writeS3(LOGGER, 'Split the data into training and test sets (.30 held out for testing)')
        (trainingData, testData) = data.randomSplit([0.7, 0.3])

        if args.train[0] != 2:
            writeS3(LOGGER, 'training RF')
            if args.numT:
                modelRF = trainModelRF(sc, trainingData, testData, LOGGER, args.numT[0])
            else:
                modelRF = trainModelRF(sc, trainingData, testData, LOGGER)

        if args.train[0] != 1:
            writeS3(LOGGER, 'training GBT')
            if args.iter:
                modelGBT = trainModelGBT(sc, trainingData, testData, LOGGER, args.iter[0])
            else:
                modelGBT = trainModelGBT(sc, trainingData, testData, LOGGER)
            

    if args.plot[0]:
        modelRF = RandomForestModel.load(sc, "s3n://bigdatap2ploans/model_2/rf")
        modelGBT = GradientBoostedTreesModel.load(sc, "s3n://bigdatap2ploans/model_2/gbt")

        writeS3(LOGGER, 'Plot RBF')
        plot(sc, modelRF, LOGGER)
        writeS3(LOGGER, 'Plot GBT')
        plot(sc, modelGBT, LOGGER)

    writeS3(LOGGER, 'Done')
