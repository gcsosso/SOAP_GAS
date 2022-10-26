import os
import pandas as pd
import numpy as np
import itertools

def getAtomSet(xyzPath):
    atomSet = set()
    for file in os.listdir(xyzPath):
        with open(xyzPath + file, 'r') as f:
            for line in f:
                if len(line.split()) == 4:
                    atomSet.add(line.split()[0].upper())
    return atomSet

def makeAtomicDataframe(xyzPath, atomSet):
    atomicDataFrame = pd.DataFrame(np.zeros((0, len(atomSet))), columns = atomSet)
    for file in os.listdir(xyzPath):
        tempDict = dict.fromkeys(atomSet, 0)
        with open(xyzPath + file, 'r') as f:
            next(f)
            next(f)
            for line in f:
                for atom in atomSet:
                    if line.split()[0].upper() == atom:
                        tempDict[line.split()[0].upper()] = 1
                        break
        atomicDataFrame = atomicDataFrame.append(tempDict, ignore_index = True)
    return atomicDataFrame

def combinationsPresent(atomicDataFrame, atomSet, maxSize):
    correctCombinations = []
    possibleCombinations = []
    for i in range(1,maxSize + 1):
       possibleCombinations += (list(itertools.combinations(atomSet,i)))
    for combo in possibleCombinations:
        test = atomicDataFrame[[i for i in combo]]
        presenceDataframe = test.loc[~(test==0).all(axis=1)]
        if presenceDataframe.shape[0] == atomicDataFrame.shape[0]:
            correctCombinations.append(combo)
    return correctCombinations

def getAtomicCounts(xyzPath, atomSet):
    atomicDataFrame = pd.DataFrame(np.zeros((0, len(atomSet))), columns = atomSet)
    tempDict = dict.fromkeys(atomSet, 0)
    for file in os.listdir(xyzPath):
        with open(xyzPath + file, 'r') as f:
            next(f)
            next(f)
            for line in f:
                for atom in atomSet:
                    if line.split()[0].upper() == atom:
                        tempDict[line.split()[0].upper()] += 1
    return dict(sorted(tempDict.items(), key=lambda item: item[1], reverse = True))

def moleculeCounts(atomicDataFrame):
    a = atomicDataFrame.sum(axis = 0).to_dict()
    for k, v in a.items():
        a[k] = int(v)
    return dict(sorted(a.items(), key=lambda item: item[1], reverse = True))

def getDropAtoms():
    data = []
    print("Which atoms (if any) would you like to drop? Please seperate them by spaces")
    try:
        data = list(map(str, input().split()))
    except ValueError:
        print("Input error")
    strData = [str(d).upper() for d in data]
    return strData

def main():
    xyzPath = str(os.path.dirname(os.path.abspath(__file__))) + "/xyz/"
    atomSet = getAtomSet(xyzPath)
    # atomSet.remove('H')
    atomCounts = getAtomicCounts(xyzPath, atomSet)
    atomicDataFrame = makeAtomicDataframe(xyzPath, atomSet)
    molecules = moleculeCounts(atomicDataFrame)
    print("The atoms present in your dataset are (excluding H): {}".format(atomSet))
    print("The atom counts are: {}".format(atomCounts))
    print("The number of molecules that contain each atom are: {}".format(molecules))
    print("The total number of molecules in the dataframe is: {}".format(atomicDataFrame.shape[0]))
    maxSize = input("What is the maximum number of atoms for the possible centre/neighbour sets? (Enter an int between 1 and {}): ".format(len(atomSet)))
    dropAtoms = getDropAtoms()
    for i in dropAtoms:
        atomSet.discard(i)
    combinations = combinationsPresent(atomicDataFrame, atomSet, int(maxSize))
    print("The combinations that you could use as centre/neighbour atoms are: {}".format(combinations))
    return 0

if __name__ == '__main__':
    main()
