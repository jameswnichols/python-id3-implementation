import math
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

def calculateClassValueEntropysFromDataset(paramClasses : list[str], dataset : list[dict], rootClass : str, pathItems : list[tuple[str, str]]):
    #Initialise a dictionary where each parameter class value has an inner dictionary with all of the root class' value counts.
    #E.g. {"boot_space": {"small":{"vgood":0,"good":4,"acc":132,"unacc":375},"med":...}, "safety" : ...}.

    #Defaultdict calls a function when the key is not found, so lambda is used to create 2 inner dictionaries with a default value of 0.
    #Defaultdicts are used as it prevents the need to check if the key is in the dictionary before adding a value to it.
    #E.g. {None : {None : 0, ...}, ...}
    paramRootValues = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    rootValueCounts = defaultdict(int)
    datasetLength = 0
    for row in dataset:
        rowItems = row.items()
        #Check if the path is completely contained in the row, with path being key-value pairs to filter by e.g. path={"buying_price":"high","safety":"med",...}.
        if pathItems <= rowItems:
            #Remove the path from the row.
            newRow = dict(rowItems - pathItems)
            for paramClass in paramClasses:
                #Increase the count of the parameter class' root class value e.g. {"boot_space": {"small":"vgood"}} += 1.
                paramRootValues[paramClass][newRow[paramClass]][newRow[rootClass]] += 1
            #Counts the root class values to calculate the entropy of the root class later.
            rootValueCounts[newRow[rootClass]] += 1
            datasetLength += 1

    paramClassEntropys = {}
    for paramClass, paramRootValueData in paramRootValues.items():
        paramEntropys = {}
        for paramValue, rootValuesDict in paramRootValueData.items():
            #Is equal to the counts of this paramter class' root class values e.g. [0, 4, 132, 375].
            rootValues = list(rootValuesDict.values())
            valTotal = sum(rootValues)
            #Calculate the entropy of each of the parameter class' values e.g. "small", "med", "big".
            valProbs = [crV / valTotal for crV in rootValues]
            paramEntropy = -sum([prob * math.log2(prob) if prob else 0 for prob in valProbs])
            paramEntropys[paramValue] = {"entropy":paramEntropy,"total":valTotal,"rootValueCounts":rootValuesDict}
        paramClassEntropys[paramClass] = paramEntropys
    
    #Calculate the entropy of the root class e.g. "quality".
    rootEntropy = -sum([((rootValueCount / datasetLength) * math.log2(rootValueCount / datasetLength)) if rootValueCount else 0 for rootValueCount in rootValueCounts.values()])

    return paramClassEntropys, rootEntropy, datasetLength
    

def calculateClassInformationGainFromDataset(classEntropys : dict, datasetLength : int, rootClassEntropy : float):
    return rootClassEntropy - sum([(cEV["total"]/datasetLength) * cEV["entropy"] for cEV in list(classEntropys.values())])

def getPossibleClassCountsFromDataset(paramClass : str, dataset : list[dict]):
    #Gets the possible values of a parameter class, and their counts e.g. "quality" = {"vgood": 54, "good":67, "acc":532, "unacc":1268}.
    possibleClasses = defaultdict(int)
    for row in dataset:
        possibleClasses[row[paramClass]] += 1
    return possibleClasses

def getLeavesBranchesFromBestClass(bestFittingClass : dict):
    leaves = []
    expand = []
    #Check the entropy of each value of the best fitting class e.g. "safety" = {"low":{"entropy":-0.0,...}, "med":{"entropy":1.23,...}, ...}.
    for value, entropyData in bestFittingClass["entropys"].items():
        if entropyData["entropy"] == 0.0:
            #If entropy of value is 0, then the value is a leaf, so find only non-zero root value.
            entropyResultValue = None
            for rootValue, rootValueCount in entropyData["rootValueCounts"].items():
                if rootValueCount > 0:
                    entropyResultValue = rootValue
                    break
            #Value is used so when validating the dataset, it can relate to a specific value of the root class e.g. "low" -> "quality" = "unacc".
            leaves.append({"value":value,"result":entropyResultValue})
        else:
            #If the entropy isnt 0, then the value is a branch, so dataset will be filtered by the value later on.
            expand.append(value)
    return leaves, expand

def renderNodes(node : Node, indent : int, rootClass : str, parentClass = None, parentClassValue = None):
    #If is used so that tree can show the previous class value of the node at the current recursion.
    #E.g. "If Safety = med; Check buying_price:".
    classValueString = f"If {parentClass} = {parentClassValue}; " if parentClassValue and parentClass else ""
    print("  " * (indent - 1) + f"{classValueString}Check {node.Class}:")
    for classValue, result in node.decisions.items():
        #E.g. "If safety = low; quality = unacc".
        print("  " * (indent) + f"If {node.Class} = {classValue}; {rootClass} = {result}")
    for pathValue, childNode in node.children.items():
        #Indent is used so that the child nodes are indented under the parent node, looks like "branches".
        renderNodes(childNode, indent + 1, rootClass,node.Class ,pathValue)

def getResultOfDatasetEntry(row : dict, startingNode : Node):
    #Instead of using recursion, uses a queue to traverse child nodes to find the tree's predicted result of a row.
    nodesToCheck = [startingNode]
    while len(nodesToCheck) > 0:
        #Get the next node in the queue.
        nodeToCheck = nodesToCheck.pop(0)
        #Get the value of the node's class in the row e.g. row["safety"] = "low".
        entryValue = row[nodeToCheck.Class]
        #If the of the node's class is a leaf, then the value of the leaf is predicted result e.g. "safety" = "low" -> "quality" = "unacc".
        if entryValue in nodeToCheck.decisions:
            return nodeToCheck.decisions[entryValue]
        #If the node's class is a branch, check that branches' node next e.g. "safety" = "med" -> "buying_price" Node.
        elif entryValue in nodeToCheck.children:
            nodesToCheck.append(nodeToCheck.children[entryValue])
    #Fallback in case the tree doesnt have a value for that row. Unacc is 70% of the dataset so it is the default result.
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

        mainClassCheckEntropy = 0
        datasetLength = 0

        classesInDataset = set(dataset[0].keys() - pathToCheck.keys())
        classesInDataset.remove(classToCheck)
        
        #By only reading the dataset rows once per while loop iteration, triples the speed of the algorithm.
        paramClassEntropys, mainClassCheckEntropy, datasetLength = calculateClassValueEntropysFromDataset(classesInDataset, dataset, classToCheck, pathItems)

        bestFittingClass = {"infoGain":-100.0}
        for paramClass, classEntropys in paramClassEntropys.items():
            classInformationGain = calculateClassInformationGainFromDataset(classEntropys, datasetLength, mainClassCheckEntropy)
            if classInformationGain > bestFittingClass["infoGain"]:
                bestFittingClass = {"class":paramClass,"infoGain":classInformationGain,"entropys":classEntropys}
        
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
    inputLength = len(fileData[1:])
    lastLine = ""
    startTime = time.time()
    for i, line in enumerate(fileData[1:]):
        print(" "*len(lastLine),end="\r")
        elapsedTime = time.time() - startTime
        lastLine = f"Time Elapsed: {round(elapsedTime,2)}s // Lines Processed: {i+1} / {inputLength} // Percentage Complete: {round((i/inputLength)*100,2)}%"
        print(lastLine,end="\r")
        dataset.append(dict(zip(classes, line.strip().split(","))))
    print(" "*len(lastLine),end="\r")
    elapsedTime = time.time() - startTime
    lastLine = f"Time Elapsed: {round(elapsedTime,2)}s // Lines Processed: {i+1} / {inputLength} // Percentage Complete: {round((i/inputLength)*100,2)}%"
    print(lastLine,end="\r")
    print("\n")
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
        if efficiency > bestTree["efficiency"] and valid / total > 0.7:
            bestTree = {"percentage":valid/total, "rootNode":rootNode, "totalNodes":totalNodes, "efficiency":efficiency}
        print(f"({run+1} / {runs}) Valid: {valid}/{total} ({round((valid/total)*100,2)}%)")
    elapsedTime = time.time() - startTime
    print(f"Best result of {runs} runs in {round(elapsedTime, 2)}s with {round((trainingSetPercentage*100) if trainingSetPercentage else 100, 2)}% of the dataset was {round(bestTree['percentage']*100,2)}% valid with an efficiency of {round(bestTree['efficiency'],2)}% and with {bestTree['totalNodes']} nodes is rendered below:")
    renderNodes(bestTree["rootNode"], 1, rootClass)
    # with open("bestTreeOutput.data", "wb") as f:
    #     pickle.dump({"Node":bestTree["rootNode"]}, f)

if __name__ == "__main__":
    loadedDataset = extractDatasetFromCSV("courseworkDataset.csv")
    rootClass = list(loadedDataset[0].keys())[-1]
    rootClassCounts = getPossibleClassCountsFromDataset(rootClass, loadedDataset)
    testFindBestTree(dataset=loadedDataset, rootClass=rootClass, rootClassCounts=rootClassCounts, runs=1500, trainingSetPercentage=None)