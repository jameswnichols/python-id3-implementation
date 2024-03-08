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
            #Value is used so when validating the dataset, it can relate to a specific value of the root class e.g. "safety" = "low" -> "quality" = "unacc".
            leaves.append({"value":value,"result":entropyResultValue})
        else:
            #If the entropy isnt 0, then the value is a branch, so dataset will be filtered by the value later on.
            expand.append(value)
    return leaves, expand

def renderNodes(node : Node, indent : int, rootClass : str, parentClass = None, parentClassValue = None):
    #Inline if is used so that tree can show the previous class value of the node at the current recursion.
    #E.g. "If Safety = med; Check buying_price:".
    classValueString = f"If {parentClass} = {parentClassValue}; " if parentClassValue and parentClass else ""
    print("  " * (indent - 1) + f"{classValueString}Check {node.Class}:")

    rootClassValuePossibles = defaultdict(list)
    for classValue, result in node.decisions.items():
        rootClassValuePossibles[result].append(classValue)
    
    for result, classValues in rootClassValuePossibles.items():
        #E.g. "If safety = low; quality = unacc".
        print("  " * (indent) + f"If {node.Class} = {', '.join(classValues)}; {rootClass} = {result}")
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
        #If the of the node's class is a leaf, then return the value as it's the predicted result e.g. "safety" = "low" -> "quality" = "unacc".
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

        #Get classes in the dataset that are not in the path e.g. "buying_price" and "boot_space" and remove root class e.g. "quality".
        #Set can be used as order of the classes is not important.
        classesInDataset = set(dataset[0].keys() - pathToCheck.keys())
        classesInDataset.remove(classToCheck)
        
        #By only reading the dataset rows once per while loop iteration it significantly increases the speed of the algorithm.
        #Doesnt necessarily help on a dataset of 1.7k rows, but adds up on larger datasets (tested on 370k rows).
        paramClassEntropys, mainClassCheckEntropy, datasetLength = calculateClassValueEntropysFromDataset(classesInDataset, dataset, classToCheck, pathItems)

        #For each of the classes in the dataset, calculate its information gain and if it is greater than the best fitting class, then set it as the best fitting class.
        bestFittingClass = {"infoGain":-100.0}
        for paramClass, classEntropys in paramClassEntropys.items():
            classInformationGain = calculateClassInformationGainFromDataset(classEntropys, datasetLength, mainClassCheckEntropy)
            if classInformationGain > bestFittingClass["infoGain"]:
                bestFittingClass = {"class":paramClass,"infoGain":classInformationGain,"entropys":classEntropys}
        
        #Set the parent node's class to the best fitting class.
        #This is done as when a node is created, it isn't assigned a class and is instead passed into the function as a parameter.
        parentNode.Class = bestFittingClass["class"]
        classValuesToLeave, classValuesToBranch = getLeavesBranchesFromBestClass(bestFittingClass)
        #Iterate over each leaf / decision of the tree.
        for classValueLeaf in classValuesToLeave:
            #Add the decision to the parent node.
            #If the parent nodees class was "safety", then the decision is "low" -> "quality" = "unacc".
            parentNode.addDecision(classValueLeaf["value"], classValueLeaf["result"])

        #Iterate over each branch of the tree.
        for classValueToBranch in classValuesToBranch:
            totalNodes += 1
            newNode = Node()
            #Add the branch to the parent node.
            #If the parent nodees class was "safety", then the branch is "med" -> "buying_price" Node.
            parentNode.addChild(classValueToBranch, newNode)
            #Copies the current path, and adds the new nodes class to it.
            pathToAdd = pathToCheck.copy()
            pathToAdd[bestFittingClass["class"]] = classValueToBranch
            pathsToCheck.append((newNode, pathToAdd))

    print(" "*len(lastLine),end="\r")
    elapsedTime = time.time() - startTime
    lastLine = f"Time Elapsed: {round(elapsedTime,2)}s // Nodes Created: {totalNodes} // Paths Checked: {valuesChecked} // Percentage Complete: {round((valuesChecked/totalNodes)*100,2)}%"
    #Using \r means that the previous line is overwritten instead of adding a new one, so that the time elapsed and progress % is updated.
    print(lastLine,end="\r")

    print("\n")
    return totalNodes, rootNode

def extractDatasetFromCSV(filename):
    dataset = []
    fileData = None
    classes = []
    with open(filename, "r") as f:
        fileData = f.readlines()
    #Classes are the headers of the csv, e.g. ["buying_price", "boot_space", "safety", "quality", ...].
    #.strip() is used to remove the newlines and any potential whitespace from the row.
    classes = fileData[0].strip().split(",")
    #The length of the file with the headers removed.
    inputLength = len(fileData[1:])
    lastLine = ""
    startTime = time.time()
    for i, line in enumerate(fileData[1:]):
        print(" "*len(lastLine),end="\r")
        elapsedTime = time.time() - startTime
        lastLine = f"Time Elapsed: {round(elapsedTime,2)}s // Lines Processed: {i+1} / {inputLength} // Percentage Complete: {round((i/inputLength)*100,2)}%"
        print(lastLine,end="\r")
        #zip creates a list of tuples, where the first value is the header and the second value is the associated value in the row.
        #E.g. [("buying_price","high"), ("boot_space","small"), ...].
        #This is then converted to a dictionary, where the first value is the key and the second value is the value.
        #E.g. {"buying_price":"high", "boot_space":"small", ...}.
        #This is then added to the dataset.
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
        #Compares the actual rows class to the predicted result of the tree.
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
    #Calculates the amount of each class value in the dataset,
    #E.g. if there are 4 root class values, then the amount of each class value is (dataset length / 4).
    datasetPerValue = (len(dataset) / len(rootClassCounts))

    for rootClassValue in rootClassCounts:
        currentOfEachClassValue[rootClassValue] = 0
        amountOfEachClassValue[rootClassValue] = math.floor(datasetPerValue * trainingPercentage)
    trainingDataset = []
    testingDataset = []
    for row in shuffleDataset:
        rowRootClassValue = row[rootClass]
        #If the current amount of that class value is less than the desired amount, then add it to the training set.
        if currentOfEachClassValue[rowRootClassValue] < amountOfEachClassValue[rowRootClassValue]:
            currentOfEachClassValue[rowRootClassValue] += 1
            trainingDataset.append(row)
        else:
            testingDataset.append(row)

    return trainingDataset, testingDataset

def testFindBestTree(dataset : list[dict], rootClass : str, rootClassCounts : list[dict[str, int]], trainingSetPercentage : float = None, runs : int = 1):
    startTime = time.time()
    bestTree = {"percentage":0, "rootNode":{}, "efficiency":-100}
    runningPercentageTotal = 0
    totalPercentageCount = 0
    for run in range(runs):
        trainingDataset, testingDataset = splitDataset(dataset, rootClass , rootClassCounts, trainingSetPercentage)
        totalNodes, rootNode = getNodesFromDataset(trainingDataset, rootClass)
        valid, total = validateDataset(testingDataset, rootNode, rootClass)
        if valid / total > bestTree["percentage"]:
            bestTree = {"percentage":valid/total, "rootNode":rootNode, "totalNodes":totalNodes}
        runningPercentageTotal += valid / total
        totalPercentageCount += 1
        print(f"({run+1} / {runs}) Valid: {valid}/{total} ({round((valid/total)*100,2)}%)")
    elapsedTime = time.time() - startTime
    averagePercentage = runningPercentageTotal / totalPercentageCount
    print(f"Best result of {runs} runs in {round(elapsedTime, 2)}s with {round((trainingSetPercentage*100) if trainingSetPercentage else 100, 2)}% of the dataset was {round(bestTree['percentage']*100,2)}% accurate with an average accuracy of {round(averagePercentage*100, 2)}% and with {bestTree['totalNodes']} nodes is rendered below:")
    renderNodes(bestTree["rootNode"], 1, rootClass)
    #Optionally comment out the line below to save the tree to a file.
    #Will need to import renderNodes from the main file if used externally.
    # with open("bestTreeOutput.data", "wb") as f:
    #     pickle.dump({"Node":bestTree["rootNode"]}, f)

if __name__ == "__main__":
    loadedDataset = extractDatasetFromCSV("courseworkDataset.csv")
    rootClass = list(loadedDataset[0].keys())[-1]
    rootClassCounts = getPossibleClassCountsFromDataset(rootClass, loadedDataset)
    testFindBestTree(dataset=loadedDataset, rootClass=rootClass, rootClassCounts=rootClassCounts, runs=10000, trainingSetPercentage=0.7)