import numpy as np
from random import sample, shuffle, random
import pandas as pd
from os.path import isfile
from ase.io import read
import os
import sys
params = __import__(sys.argv[1])
from quippy import descriptors
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import pickle as pkl
import functools
import time
import concurrent.futures
import functools
import operator

class individualDescriptor:
    def __init__(self, cutoff, l_max, n_max, sigma, lower, upper, mu, mu_hat, nu, nu_hat, centres, neighbours, average, mutationChance, min_cutoff, max_cutoff, min_sigma, max_sigma):
        self.cutoff = cutoff
        self.l_max = l_max
        self.n_max = n_max
        self.sigma = sigma
        self.lower = lower
        self.upper = upper
        self.min_cutoff = min_cutoff
        self.max_cutoff = max_cutoff
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.nu = nu
        self.nu_hat = nu_hat
        self.mu = mu
        self.mu_hat = mu_hat
        self.centres = centres
        self.neighbours = neighbours
        self.average = average
        self.mutationChance = mutationChance
        self.shortDescVector = [cutoff, l_max, n_max, sigma]

    def mutate(self):
        randomVals = np.random.randint(self.lower, self.upper, 2)
        cutoffMutated = np.random.randint(self.min_cutoff, self.max_cutoff)
        l_maxMutated, n_maxMutated = randomVals[0], randomVals[1]
        sigmaMutated = round(np.random.uniform(self.min_sigma, self.max_sigma), 2)
        mutatedVals = [cutoffMutated, l_maxMutated, n_maxMutated, sigmaMutated]
        prob = np.random.rand(4)
        if prob[0] > self.mutationChance:
            mutatedVals[0] = self.cutoff
        if prob[1] > self.mutationChance:
            mutatedVals[1] = self.l_max
        if prob[2] > self.mutationChance:
            mutatedVals[2] = self.n_max
        if prob[3] > self.mutationChance:
            mutatedVals[3] = self.sigma
        return individualDescriptor(mutatedVals[0], mutatedVals[1], mutatedVals[2], mutatedVals[3], self.lower, self.upper, self.mu, self.mu_hat, self.nu,\
        self.nu_hat, self.centres, self.neighbours, self.average, self.mutationChance, self.min_cutoff, self.max_cutoff, self.min_sigma, self.max_sigma)

class individual:
    def __init__(self, individualDescriptorList):
        self.individualDescriptorList = individualDescriptorList
        self.descList = [getDescriptorString(individualDescriptor) for individualDescriptor in individualDescriptorList]
        print("Getting SOAPS for {}".format(self.descList))
        start = time.time()
        try:
            self.SOAPS, self.targets = getSoaps(self.descList)
            print("Starting ML for {}".format(self.descList))
            results = getScoreSimpleRegression(self.SOAPS, self.targets)
            self.score, self.X_train, self.X_test, self.y_train, self.y_test, self.y_test_pred, self.y_train_pred = np.average(results[0]),results[1],results[2],results[3],results[4],results[5], results[6]
            print("SCORE: {}".format(self.score))
        except Exception as e:
            self.score = 9999
            self.X_train, self.X_test, self.y_train, self.y_test, self.y_test_pred, self.y_train_pred = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            print(e)
        end = time.time()
        self.timeTaken = end-start

def generateRandomValues(descDict):
    randomVals = np.random.randint(descDict['lower'], descDict['upper'], 2)
    cutoff = np.random.randint(descDict['min_cutoff'], descDict['max_cutoff'])
    l_max, n_max = randomVals[0], randomVals[1]
    sigma = round(np.random.uniform(descDict['min_sigma'], descDict['max_sigma']), 2)
    return individualDescriptor(cutoff, l_max, n_max, sigma, descDict['lower'], descDict['upper'], descDict['mu'], descDict['mu_hat'], descDict['nu'], descDict['nu_hat'],\
      descDict['centres'], descDict['neighbours'], descDict['average'], descDict['mutationChance'],descDict['min_cutoff'], descDict['max_cutoff'],\
     descDict['min_sigma'], descDict['max_sigma'])

def initialiseIndividuals(descList, popSize):
    writeToFile("Generating random values for initial population")
    return [[generateRandomValues(descDict) for descDict in descList] for _ in range(popSize)]

def getLenAtoms(s):
    return str(len(s.split()))

def nextGeneration(previousGeneration, best, lucky, numberChildren):
    nextBreeders = sorted(previousGeneration, key=lambda x: x.score, reverse=False)
    parents = selectFromPopulation(nextBreeders, best, lucky)
    parentPairings = pairwise(parents, numberChildren)
    zippedDescriptors = [list(zip(pairing[0].individualDescriptorList, pairing[1].individualDescriptorList)) for pairing in parentPairings]
    return [haveOffspringParent(zippedDescriptor) for zippedDescriptor in zippedDescriptors]

def selectFromPopulation(sortedPopulationList, bestSample, luckyFew):
    bestIndividuals = sortedPopulationList[:bestSample]
    luckyIndividuals = sample(sortedPopulationList[bestSample:], luckyFew)
    writeToFile("The best individuals are:")
    for ind in bestIndividuals:
        writeToFile("{} with a score of: {}".format(ind.descList, ind.score))
    writeToFile("The lucky individuals are:")
    for ind in luckyIndividuals:
        writeToFile("{} with a score of: {}".format(ind.descList, ind.score))
    return bestIndividuals + luckyIndividuals

def getPopulation(initialPopulation):
    return [individual(individualDescriptorList) for individualDescriptorList in initialPopulation]

def getDescriptorString(individualDescriptor):
    if individualDescriptor.average == True:
        avg = " average "
    else:
        avg = " "

    return "soap{}cutoff={} l_max={} n_max={} atom_sigma={} n_Z={} Z={} n_species={} species_Z={} mu={} mu_hat={} nu={} nu_hat={}".format(avg,\
        individualDescriptor.cutoff, individualDescriptor.l_max, individualDescriptor.n_max, individualDescriptor.sigma, getLenAtoms(individualDescriptor.centres),\
            individualDescriptor.centres, getLenAtoms(individualDescriptor.neighbours), individualDescriptor.neighbours, individualDescriptor.mu, individualDescriptor.mu_hat, individualDescriptor.nu, individualDescriptor.nu_hat)

def pairwise(t, numberChildren):
    l = []
    for i in range(numberChildren):
        shuffle(t)
        it = iter(t)
        l += list(zip(it,it))
    return l

def coinFlip(option1, option2):
    np.random.seed()
    randomValue = np.random.random()
    if randomValue >= 0.5:
        return option1
    else:
        return option2

def haveOffspringDescriptor(individualDescriptorTuple):
    if individualDescriptorTuple[0].centres != individualDescriptorTuple[1].centres:
        print("Centres are not the same")
    if individualDescriptorTuple[0].neighbours != individualDescriptorTuple[1].neighbours:
        print("Neighbours are not the same")
    if individualDescriptorTuple[0].average != individualDescriptorTuple[1].average:
        print("Average is not the same")
    output = individualDescriptor(coinFlip(individualDescriptorTuple[0].cutoff, individualDescriptorTuple[1].cutoff), coinFlip(individualDescriptorTuple[0].l_max,\
        individualDescriptorTuple[1].l_max), coinFlip(individualDescriptorTuple[0].n_max, individualDescriptorTuple[1].n_max), coinFlip(individualDescriptorTuple[0].sigma,\
        individualDescriptorTuple[1].sigma),individualDescriptorTuple[0].lower, individualDescriptorTuple[0].upper, individualDescriptorTuple[0].mu, individualDescriptorTuple[0].mu_hat,\
        individualDescriptorTuple[0].nu, individualDescriptorTuple[0].nu_hat, individualDescriptorTuple[0].centres, individualDescriptorTuple[0].neighbours, \
        individualDescriptorTuple[0].average, individualDescriptorTuple[0].mutationChance, individualDescriptorTuple[0].min_cutoff, individualDescriptorTuple[0].max_cutoff, \
        individualDescriptorTuple[0].min_sigma, individualDescriptorTuple[0].max_sigma).mutate()
    return output

def haveOffspringParent(zippedDescriptorTuple):
    return individual([haveOffspringDescriptor(individualDescriptorTuple) for individualDescriptorTuple in list(zippedDescriptorTuple)])


def createChildren(nextBreeders, numberChildren):
    nextGeneration = []
    shuffle(nextBreeders)
    parentPairings = pairwise(nextBreeders)
    for parent in parentPairings:
        list(zip(parent[0].individualDescriptorList, parent[1].individualDescriptorList))

def getPopulation(initialPopulation):
    return [individual(individualDescriptorList) for individualDescriptorList in initialPopulation]

def multipleGeneration(descList, numberOfGenerations, popSize, bestSample, luckyFew, numberChildren):
    filename = str(os.path.dirname(os.path.abspath(__file__))) + "/history_{}.pkl".format(sys.argv[1])
    with open(filename, 'ab+') as fp:
        history = []
        initPop = initialiseIndividuals(descList, popSize)
        history.append(getPopulation(initPop))
        pkl.dump(sorted(history[0],key=lambda x: x.score), fp)
        for i in range(numberOfGenerations):
            print("\n Generation:{}".format(i))
            writeToFile("\n Generation:{}".format(i))
            ########## TIMING ##########
            totalTimeTaken = 0
            for ind in history[0]:
                totalTimeTaken += ind.timeTaken
            writeToFile("\nTime taken for this generation: {}".format(totalTimeTaken))
            nextGen = nextGeneration(history[0], bestSample, luckyFew, numberChildren)
            history.append(nextGen)
            pkl.dump(sorted(history[1],key=lambda x: x.score), fp)
            history.pop(0)
    return history

def getSoaps(descList):
    databasePath = str(os.path.dirname(os.path.abspath(__file__))) + "/database.csv"
    db = pd.read_csv(databasePath, index_col = 'Name')
    db.dropna(inplace = True)
    compoundNames = db.index.tolist()
    savedData = []
    savedTargets = []
    for name in compoundNames:
        soap = []
        file = str(os.path.dirname(os.path.abspath(__file__))) + "/xyz/{}.xyz".format(name)
        assert isfile(file), "File {} doesn't exist or isn't readable. Please read the README".format(file)
        at = read(file)
        savedTargets.append(db.loc[name]['Target'])
        for desc in descList:
            soap+=list(descriptors.Descriptor(desc).calc(at)['data'][0])
        savedData.append(soap)
    return np.array(savedData, dtype=object), np.array(savedTargets, dtype=object)

def algorithmScoreRegression(descList):
    soaps, targets = getSoaps(descList)
    score = getScoreNoCV(soaps, targets)
    #### For MSE
    return score * -1


def getScoreNoCV(soaps, targets):
    y=targets
    X = soaps
    scaler = MaxAbsScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    estimator = RandomForestRegressor(random_state=0)
    return scorer(estimator, X_train, X_test, y_train, y_test)

def crossValidation(cv, X, y):
    score, X_train_res, X_test_res, y_train_res, y_test_res, y_test_pred, y_train_pred = [], [], [], [], [], [], []
    estimator = RandomForestRegressor(max_depth=6, random_state=0)
    for train_index, test_index in cv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        res = scorer(estimator, X_train, X_test, y_train, y_test)
        score.append(res[0])
        X_train_res.append([res[1]])
        X_test_res.append([res[2]])
        y_train_res.append([res[3]])
        y_test_res.append([res[4]])
        y_test_pred.append([res[5]])
        y_train_pred.append([res[6]])
    return [score, X_train_res, X_test_res, y_train_res, y_test_res, y_test_pred, y_train_pred]

def scorer(estimator, X_train, X_test, y_train, y_test):
    estimator.fit(X_train, y_train)
    y_test_pred, y_train_pred = estimator.predict(X_test), estimator.predict(X_train)
    y_test = np.ravel(y_test)
    y_train = np.ravel(y_train)
    testCorr = pearsonr(y_test, y_test_pred)[0]
    trainCorr = pearsonr(y_train, y_train_pred)[0]
    testMSE = mean_squared_error(y_test, y_test_pred)
    trainMSE =  mean_squared_error(y_train, y_train_pred)
    return (2 * (trainMSE * (1-trainCorr)) + (testMSE * (1-testCorr))), X_train, X_test, y_train, y_test, y_test_pred, y_train_pred

def getScoreSimpleRegression(soaps, targets):
    cv = KFold(n_splits=5, shuffle = True, random_state=0)
    return crossValidation(cv, soaps, targets)

def writeToFile(words):
    outputFile = str(os.path.dirname(os.path.abspath(__file__))) + "/out_{}.txt".format(sys.argv[1])
    with open(outputFile, 'a+') as f:
        f.write(words)
        f.write('\n')
    return

def check_file(filePath):
    if os.path.exists(filePath):
        print("Making Backup for {}".format(filePath))
        numb = 1
        while True:
            newPath = "{}-Backup{}.pkl".format(filePath[:-4], numb)
            if os.path.exists(newPath):
                numb += 1
            else:
                break
        os.rename(filePath, newPath)
        print("Backed up {} to {}".format(filePath, newPath))
        return

def checkCorrectSizes(popSize, bestSample, luckyFew, numberChildren):
    if ((bestSample + luckyFew) / 2 * numberChildren != popSize):
        print("population size not stable")
        return False
    return True

def readBestHistory():
    data = []
    with open(str(os.path.dirname(os.path.abspath(__file__))) + "/history_{}.pkl".format(sys.argv[1]), 'rb') as fr:
        try:
            while True:
                ind = pkl.load(fr)[0]
                ind.SOAPS, ind.targets, ind.X_train, ind.X_test = np.nan, np.nan, np.nan, np.nan
                data.append(ind)
        except EOFError:
            pass
    return data

def Main(params):
    numberOfGenerations = params.numberOfGenerations
    popSize = params.popSize
    bestSample = params.bestSample
    luckyFew = params.luckyFew
    numberChildren = params.numberChildren
    descList = params.descList
    print(descList)
    if checkCorrectSizes(popSize, bestSample, luckyFew, numberChildren) == False:
        return 1
    check_file(str(os.path.dirname(os.path.abspath(__file__))) + "/history_{}.pkl".format(sys.argv[1]))
    check_file(str(os.path.dirname(os.path.abspath(__file__))) + "/out_{}.txt".format(sys.argv[1]))
    writeToFile("Starting genetic algorithm with the following parameters:")
    writeToFile("Population size: {}".format(popSize))
    writeToFile("Sample size of best individuals: {}".format(bestSample))
    writeToFile("Sample size of lucky individuals: {}".format(luckyFew))
    writeToFile("Number of children: {}".format(numberChildren))
    writeToFile("The descriptor parameters being used are: {}".format(descList))
    history = multipleGeneration(descList, numberOfGenerations, popSize, bestSample, luckyFew, numberChildren)
    writeToFile("Finished! \n")
    bestHistory = readBestHistory()
    writeToFile("The best individual was: {} \n with a score of: {}".format(bestHistory[0].descList, bestHistory[0].score))
    pkl.dump(bestHistory, open(str(os.path.dirname(os.path.abspath(__file__))) + "/best_{}.pkl".format(sys.argv[1]), "wb"))
    return

if __name__ == '__main__':
    start = time.time()
    Main(params)
    end = time.time()
    writeToFile("Time taken: {}s".format(end-start)) 
