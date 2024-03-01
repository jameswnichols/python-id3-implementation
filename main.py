import math
import copy
import pickle
import time
import random

class Node:
    def __init__(self, classValue):
        self.children = {}
        self.decisions = {}
        self.Class = classValue
    
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

def calculateClassValueEntropyFromDataset(paramClass : str, dataset : list[dict], checkClass : str):
    #Initialise a dictionary where each parameter class value has an inner dictionary with all of the root class' value counts.
    #E.g. for "boot_space": {"small":{"vgood":0,"good":4,"acc":132,"unacc":375},"med":...}
    paramRootValues = {}
    for row in dataset:

        #Increase the count of the parameter class' root class value, or create a new entry if it doesn't exist.
        if row[paramClass] not in paramRootValues:
            paramRootValues[row[paramClass]] = {}

        if row[checkClass] not in paramRootValues[row[paramClass]]:
            paramRootValues[row[paramClass]][row[checkClass]] = 1
        else:
            paramRootValues[row[paramClass]][row[checkClass]] += 1
        
    paramEntropys = {}

    for paramValue, rootValuesDict in paramRootValues.items():
        rootValues = list(rootValuesDict.values())
        #Calculate the entropy of each of the parameter class' values.
        valTotal = sum([crV for crV in rootValues])
        valProbs = [crV / valTotal for crV in rootValues]
        paramEntropy = -sum([prob * math.log2(prob) if prob else 0 for prob in valProbs])
        paramEntropys[paramValue] = {"entropy":paramEntropy,"total":valTotal,"results":rootValuesDict}
    return paramEntropys

def calculateClassInformationGainFromDataset(classEntropys : dict, dataset : list[dict], checkClassEntropy : float):
    return checkClassEntropy - sum([(cEV["total"]/len(dataset)) * cEV["entropy"] for cEV in list(classEntropys.values())])

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

def getPossibleClassValuesFromDataset(paramClass : str, dataset : list[dict]):
    possibleClasses = []
    for row in dataset:
        possibleClasses.append(row[paramClass])
    return list(dict.fromkeys(possibleClasses))

def getValuesLeavesExpandFromBestClass(bestFittingClass : dict):
    leaves = []
    expand = []
    #Check if multiple values result in the same thing, if so then group them.
    for value, entropyData in bestFittingClass["entropys"].items():
        if entropyData["entropy"] == 0.0:
            entropyResultValue = [rK for rK, rV in entropyData["results"].items() if rV > 0][0]
            #allLeafValues.extend(entropyResultValue)
            leaves.append({"value":value,"result":entropyResultValue})
        else:
            expand.append(value)
    return leaves, expand

def renderNodes(node : Node, indent : int, checkClassName : str, parentClass = None, parentClassValue = None):
    classValueString = f"If {parentClass} = {parentClassValue}; " if parentClassValue and parentClass else ""
    print("  " * (indent - 1) + f"{classValueString}Check {node.Class}:")
    for classValue, result in node.decisions.items():
        print("  " * (indent) + f"If {node.Class} = {classValue}; {checkClassName} = {result}")
    for pathValue, childNode in node.children.items():
        renderNodes(childNode, indent + 1, checkClassName,node.Class ,pathValue)

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

def getKeyFromPath(path : list[dict]):
    nodeClassPathList = []
    for innerPath in path:
        nodePathKey, nodePathValue = list(innerPath.keys())[0], list(innerPath.values())[0]
        nodeClassPathList.append(f"{nodePathKey}:{nodePathValue}")
    return "/".join(nodeClassPathList)

def getNodesFromDataset(dataset : list[dict], classToCheck : str):
    #classToCheckValues = getPossibleClassValuesFromDataset(classToCheck, dataset)

    rootNode = Node("Root")
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

        currentDataset = dataset
        parentNode, pathToCheck = pathsToCheck.pop(0)

        #For each parameter value, get its unique values, e.g. {"buying_price":["vhigh", "high", "med", "low"],"safety":...}
        paramUniqueValues = {}
        #For each parameter value, get the number of unique values, e.g. {"buying_price":{"vhigh":10,"high":45,...},"safety":...}
        paramCounts = {}

        currentDataset = filterDataset(currentDataset, pathToCheck)
        
        mainClassCheckEntropy = calculateEntropyFromDataset(classToCheck, currentDataset)

        classesToCheck = [x for x in list(currentDataset[0].keys()) if x != classToCheck]

        bestFittingClass = {"infoGain":-100.0}
        for dClass in classesToCheck:
            classEntropys = calculateClassValueEntropyFromDataset(dClass, currentDataset, classToCheck)
            classInformationGain = calculateClassInformationGainFromDataset(classEntropys, currentDataset, mainClassCheckEntropy)
            if classInformationGain > bestFittingClass["infoGain"]:
                bestFittingClass = {"class":dClass,"infoGain":classInformationGain,"entropys":classEntropys}
        
        parentNode.Class = bestFittingClass["class"]
        classValueLeaves, classValuesToExpand = getValuesLeavesExpandFromBestClass(bestFittingClass)
        for classValueLeaf in classValueLeaves:
            parentNode.addDecision(classValueLeaf["value"], classValueLeaf["result"])

        for classValueToExpand in classValuesToExpand:
            totalNodes += 1
            newNode = Node("NaN")
            parentNode.addChild(classValueToExpand, newNode)
            pathToAdd = pathToCheck.copy()
            pathToAdd[bestFittingClass["class"]] = classValueToExpand
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

def validateDataset(dataset : list[dict], rootNode : dict[Node], checkClass : str):
    valid = 0
    for row in dataset:
        if row[checkClass] == getResultOfDatasetEntry(row, rootNode):
            valid += 1
    return valid, len(dataset)

def splitDataset(dataset : list[dict], targetClass : str, trainingPercentage : float = None):
    shuffleDataset = dataset.copy()
    random.shuffle(shuffleDataset)
    if not trainingPercentage:
        return shuffleDataset, dataset
    targetClassColumn = [row[targetClass] for row in dataset]
    targetClassValues = list(dict.fromkeys(targetClassColumn))
    targetClassValueRatios = {targetClassValue : targetClassColumn.count(targetClassValue)/len(dataset) for targetClassValue in targetClassValues}
    amountOfEach = {targetClassValue : math.floor(targetClassValueRatio * len(dataset) * trainingPercentage) for targetClassValue, targetClassValueRatio in targetClassValueRatios.items()}
    currentOfEach = {targetClassValue : 0 for targetClassValue in targetClassValues}
    trainingDataset = []
    testingDataset = []
    for row in shuffleDataset:
        rowTargetClass = row[targetClass]
        if currentOfEach[rowTargetClass] < amountOfEach[rowTargetClass]:
            trainingDataset.append(row)
            currentOfEach[rowTargetClass] += 1
        else:
            testingDataset.append(row)

    return trainingDataset, testingDataset

def testFindBestTree(dataset : list[dict], checkClass : str, trainingSetPercentage : float = None, runs : int = 1):
    startTime = time.time()
    bestTree = {"percentage":0, "rootNode":{}, "accuracy":-100}
    for run in range(runs):
        trainingDataset, testingDataset = splitDataset(dataset, checkClass, trainingSetPercentage)
        totalNodes, rootNode = getNodesFromDataset(trainingDataset, checkClass)
        valid, total = validateDataset(testingDataset, rootNode, checkClass)
        accuracy = ((valid/total) * 100) / math.log2(totalNodes)
        if accuracy > bestTree["accuracy"] and valid / total > 0.8:
            bestTree = {"percentage":valid/total, "rootNode":rootNode, "totalNodes":totalNodes, "accuracy":accuracy}
        print(f"({run+1} / {runs}) Valid: {valid}/{total} ({round((valid/total)*100,2)}%)")
    elapsedTime = time.time() - startTime
    print(f"Best result of {runs} runs in {round(elapsedTime, 2)}s with {round((trainingSetPercentage*100) if trainingSetPercentage else 100, 2)}% of the dataset was {round(bestTree['percentage']*100,2)}% valid with an efficiency of {round(bestTree['accuracy'],2)}% and with {bestTree['totalNodes']} nodes, rendered below:")
    renderNodes(bestTree["rootNode"], 1, checkClass)

if __name__ == "__main__":
    loadedDataset = extractDatasetFromCSV("courseworkDataset.csv")
    checkClass = list(loadedDataset[0].keys())[-1]
    #nodes = getNodesFromDataset(dataset=loadedDataset, classToCheck=checkClass)
    #performanceTest(dataset=loadedDataset, checkClass=checkClass, trainingSetPercentage=0.8)
    # for i in range(0, 100):
    testFindBestTree(dataset=loadedDataset, checkClass=checkClass, runs=100, trainingSetPercentage=None)