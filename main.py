import math
import copy
import pickle
import time
import random
from collections import defaultdict

class Node:
    def __init__(self):
        self.children = {}
        self.decisions = {}
        self.Class = "NaN"
    
    def addChild(self, value, node):
        self.children[value] = node

    def addDecision(self, classValue, result):
        self.decisions [classValue] = result


def calculateEntropyFromDataset(parameter : str, dataset : list[dict]):
    probabilities = {}
    #Get every value in the dataset from the corosponding parameter.
    paramValues = [row[parameter] for row in dataset]
    #list(dict.fromkeys(x)) removes duplicate values while preserving order.
    for x in list(dict.fromkeys(paramValues)):
        #Calculate the probability of each unique value.
        probabilities[x] = paramValues.count(x)/len(dataset)

    #Calculate the entropy of the value.
    return -sum([prob * math.log2(prob) for prob in probabilities.values()])

def calculateClassValueEntropyFromDataset(paramClass : str, dataset : list[dict], rootClass : str, pathItems : list[tuple[str, str]], calculateRootEntropy : bool = False):
    #Initialise a dictionary where each parameter class value has an inner dictionary with all of the root class' value counts.
    #E.g. for "boot_space": {"small":{"vgood":0,"good":4,"acc":132,"unacc":375},"med":...}

    #Defaultdict calls a function when the key is not found, so lambda is used to create an inner dictionary with a default value of 0.
    paramRootValues = defaultdict(lambda: defaultdict(int))
    rootValueCounts = defaultdict(int)
    datasetLength = 0
    for row in dataset:
        rowItems = row.items()
        if pathItems <= rowItems:
            newRow = dict(rowItems - pathItems)
            #Increase the count of the parameter class' root class value.            
            paramRootValues[newRow[paramClass]][newRow[rootClass]] += 1

            if calculateRootEntropy:
                rootValueCounts[newRow[rootClass]] += 1
            datasetLength += 1
    paramEntropys = {}

    for paramValue, rootValuesDict in paramRootValues.items():
        rootValues = list(rootValuesDict.values())
        #Calculate the entropy of each of the parameter class' values.
        valTotal = sum(rootValues)
        valProbs = [crV / valTotal for crV in rootValues]
        paramEntropy = -sum([prob * math.log2(prob) if prob else 0 for prob in valProbs])
        paramEntropys[paramValue] = {"entropy":paramEntropy,"total":valTotal,"rootValueCounts":rootValuesDict}

    if calculateRootEntropy:
        rootEntropy = -sum([((rootValueCount / datasetLength) * math.log2(rootValueCount / datasetLength)) if rootValueCount else 0 for rootValueCount in rootValueCounts.values()])
        return paramEntropys, rootEntropy, datasetLength
    return paramEntropys

def calculateClassInformationGainFromDataset(classEntropys : dict, datasetLength : int, rootClassEntropy : float):
    return rootClassEntropy - sum([(cEV["total"]/datasetLength) * cEV["entropy"] for cEV in list(classEntropys.values())])

def filterDataset(dataset : list[dict], path : dict):
    newDataset = []
    pathItems = path.items()
    for row in dataset:
        rowItems = row.items()
        #If the row contains all the values in the path e.g. path={"buying_price":"high","boot_space":"small",...} then keep it.
        if pathItems <= rowItems:
            #Remove the path from the row.
            newRow = dict(rowItems - pathItems)
            newDataset.append(newRow)
    return newDataset

def getPossibleClassCountsFromDataset(paramClass : str, dataset : list[dict]):
    possibleClasses = {}
    for row in dataset:
        if row[paramClass] not in possibleClasses:
            possibleClasses[row[paramClass]] = 1
        else:
            possibleClasses[row[paramClass]] += 1
        #possibleClasses.append(row[paramClass])
    return possibleClasses

def getLeavesBranchesFromBestClass(bestFittingClass : dict):
    leaves = []
    expand = []
    #Check if multiple values result in the same thing, if so then group them.
    for value, entropyData in bestFittingClass["entropys"].items():
        if entropyData["entropy"] == 0.0:
            #If entropy of value is 0, then the value is a leaf, so find only non-zero root value.
            entropyResultValue = None
            for rootValue, rootValueCount in entropyData["rootValueCounts"].items():
                if rootValueCount > 0:
                    entropyResultValue = rootValue
                    break

            leaves.append({"value":value,"result":entropyResultValue})
        else:
            expand.append(value)
    return leaves, expand

def renderNodes(node : Node, indent : int, rootClass : str, parentClass = None, parentClassValue = None):
    classValueString = f"If {parentClass} = {parentClassValue}; " if parentClassValue and parentClass else ""
    print("  " * (indent - 1) + f"{classValueString}Check {node.Class}:")
    for classValue, result in node.decisions.items():
        print("  " * (indent) + f"If {node.Class} = {classValue}; {rootClass} = {result}")
    for pathValue, childNode in node.children.items():
        renderNodes(childNode, indent + 1, rootClass,node.Class ,pathValue)

def getResultOfDatasetEntry(entry : dict, startingNode : Node):
    nodesToCheck = [startingNode]
    while len(nodesToCheck) > 0:
        nodeToCheck = nodesToCheck.pop(0)
        entryValue = entry[nodeToCheck.Class]
        if entryValue in nodeToCheck.decisions:
            return nodeToCheck.decisions[entryValue]
        elif entryValue in nodeToCheck.children:
            nodesToCheck.append(nodeToCheck.children[entryValue])

    return "unacc"

def getNodesFromDataset(dataset : list[dict], classToCheck : str):
    rootNode = Node()
    pathsToCheck = [(rootNode, {})]
    totalNodes = 1
    valuesChecked = 0

    startTime = time.time()
    lastLine = ""
    while len(pathsToCheck) > 0:
        print(" "*len(lastLine),end="\r")
        elapsedTime = time.time() - startTime
        lastLine = f"Time Elapsed: {round(elapsedTime,2)}s // Nodes Created: {totalNodes} // Paths Checked: {valuesChecked} // Percentage Complete: {round((valuesChecked/totalNodes)*100,2)}%"
        print(lastLine,end="\r")
        valuesChecked += 1

        #Path to check is a dictionary of class:value pairs to remove e.g. path={"buying_price":"high","boot_space":"small",...}
        parentNode, pathToCheck = pathsToCheck.pop(0)
        pathItems = pathToCheck.items()

        calculatedMainClassEntropy = False
        mainClassCheckEntropy = 0
        datasetLength = 0

        classesInDataset = set(dataset[0].keys() - pathToCheck.keys())
        classesInDataset.remove(classToCheck)

        bestFittingClass = {"infoGain":-100.0}
        for classInDataset in classesInDataset:
            if not calculatedMainClassEntropy:
                #By only reading the dataset rows once per while loop iteration, triples the speed of the algorithm.
                classEntropys, mainClassCheckEntropy, datasetLength = calculateClassValueEntropyFromDataset(classInDataset, dataset, classToCheck, pathItems, True)
                calculatedMainClassEntropy = True
            else:
                classEntropys = calculateClassValueEntropyFromDataset(classInDataset, dataset, classToCheck, pathItems)

            classInformationGain = calculateClassInformationGainFromDataset(classEntropys, datasetLength, mainClassCheckEntropy)
            if classInformationGain > bestFittingClass["infoGain"]:
                bestFittingClass = {"class":classInDataset,"infoGain":classInformationGain,"entropys":classEntropys}
        
        parentNode.Class = bestFittingClass["class"]
        classValuesToLeave, classValuesToBranch = getLeavesBranchesFromBestClass(bestFittingClass)
        for classValueLeaf in classValuesToLeave:
            parentNode.addDecision(classValueLeaf["value"], classValueLeaf["result"])

        for classValueToBranch in classValuesToBranch:
            totalNodes += 1
            newNode = Node()
            parentNode.addChild(classValueToBranch, newNode)
            pathToAdd = pathToCheck.copy()
            pathToAdd[bestFittingClass["class"]] = classValueToBranch
            pathsToCheck.append((newNode, pathToAdd))

    print(" "*len(lastLine),end="\r")
    elapsedTime = time.time() - startTime
    lastLine = f"Time Elapsed: {round(elapsedTime,2)}s // Nodes Created: {totalNodes} // Paths Checked: {valuesChecked} // Percentage Complete: {round((valuesChecked/totalNodes)*100,2)}%"
    print(lastLine,end="\r")

    print("\n")
    return totalNodes, rootNode

def extractDatasetFromCSV(filename):
    dataset = []
    fileData = None
    classes = []
    with open(filename, "r") as f:
        fileData = f.readlines()
    classes = fileData[0].strip().split(",")
    for line in fileData[1:]:
        lineData = {}
        for i, value in enumerate(line.strip().split(",")):
            lineData[classes[i]] = value    
        dataset.append(lineData)
    return dataset

def validateDataset(dataset : list[dict], rootNode : dict[Node], rootClass : str):
    valid = 0
    for row in dataset:
        if row[rootClass] == getResultOfDatasetEntry(row, rootNode):
            valid += 1
    return valid, len(dataset)

def splitDataset(dataset : list[dict], rootClass : str, rootClassCounts : list[str], trainingPercentage : float = None):
    shuffleDataset = dataset.copy()
    random.shuffle(shuffleDataset)

    #If using the whole dataset to train, then testing set will be whole dataset.
    if not trainingPercentage:
        return shuffleDataset, dataset
    
    amountOfEachClassValue = {}
    currentOfEachClassValue = {}
    for rootClassValue, rootClassCount in rootClassCounts.items():
        currentOfEachClassValue[rootClassValue] = 0
        amountOfEachClassValue[rootClassValue] = math.floor(rootClassCount * trainingPercentage)
     
    trainingDataset = []
    testingDataset = []
    for row in shuffleDataset:
        rowRootClassValue = row[rootClass]
        if currentOfEachClassValue[rowRootClassValue] < amountOfEachClassValue[rowRootClassValue]:
            currentOfEachClassValue[rowRootClassValue] += 1
            trainingDataset.append(row)
        else:
            testingDataset.append(row)

    return trainingDataset, testingDataset

def testFindBestTree(dataset : list[dict], rootClass : str, rootClassCounts : list[dict[str, int]], trainingSetPercentage : float = None, runs : int = 1):
    startTime = time.time()
    bestTree = {"percentage":0, "rootNode":{}, "efficiency":-100}
    for run in range(runs):
        trainingDataset, testingDataset = splitDataset(dataset, rootClass , rootClassCounts, trainingSetPercentage)
        totalNodes, rootNode = getNodesFromDataset(trainingDataset, rootClass)
        valid, total = validateDataset(testingDataset, rootNode, rootClass)
        efficiency = ((valid/total) * 100) / math.log2(totalNodes)
        if efficiency > bestTree["efficiency"] and valid / total > 0.8:
            bestTree = {"percentage":valid/total, "rootNode":rootNode, "totalNodes":totalNodes, "efficiency":efficiency}
        print(f"({run+1} / {runs}) Valid: {valid}/{total} ({round((valid/total)*100,2)}%)")
    elapsedTime = time.time() - startTime
    print(f"Best result of {runs} runs in {round(elapsedTime, 2)}s with {round((trainingSetPercentage*100) if trainingSetPercentage else 100, 2)}% of the dataset was {round(bestTree['percentage']*100,2)}% valid with an efficiency of {round(bestTree['efficiency'],2)}% and with {bestTree['totalNodes']} nodes, rendered below:")
    renderNodes(bestTree["rootNode"], 1, rootClass)

if __name__ == "__main__":
    loadedDataset = extractDatasetFromCSV("courseworkDataset.csv")
    rootClass = list(loadedDataset[0].keys())[-1]
    rootClassCounts = getPossibleClassCountsFromDataset(rootClass, loadedDataset)
    testFindBestTree(dataset=loadedDataset, rootClass=rootClass, rootClassCounts=rootClassCounts, runs=1, trainingSetPercentage=None)