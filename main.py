import math
import copy
import pickle
import time
import random
import math
import matplotlib.pyplot as plt
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
        probabilities[x] = paramValues.count(x)/len(paramValues)
    #Calculate the entropy of the value.
    return -sum([prob * math.log2(prob) for prob in probabilities.values()])

def calculateClassValueEntropyFromDataset(paramClass : str, dataset : list[dict], checkClass : str, checkClassValues : list[str]):
    #Get all unique values for the parameter class.
    paramValues = list(dict.fromkeys([row[paramClass] for row in dataset]))
    #Construct a dictionary where each parameter class value has an inner dictionary with each unique root class value and a count.
    paramProb = {paramValue : {classValue : 0 for classValue in checkClassValues} for paramValue in paramValues}
    for row in dataset:
        #Increase the count of the parameter class' root class value.
        paramProb[row[paramClass]][row[checkClass]] += 1
    paramEntropys = {}
    for paramVal, checkResults in paramProb.items():
        valTotal = sum(crV for crV in list(checkResults.values()))
        valProbs = [crV / valTotal for crV in list(checkResults.values())]
        paramEntropy = -sum([prob * math.log2(prob) if prob else 0 for prob in valProbs])
        paramEntropys[paramVal] = {"entropy":paramEntropy,"total":valTotal,"results":checkResults}
    return paramEntropys

def calculateClassInformationGainFromDataset(classEntropys : dict, dataset : list[dict], checkClassEntropy : float):
    return checkClassEntropy - sum([(cEV["total"]/len(dataset)) * cEV["entropy"] for cEV in list(classEntropys.values())])

def filterDataset(dataset : list[dict], path : dict):
    pathKey, pathValue = list(path.items())[0]
    newDataset = []
    for row in dataset:
        if pathKey in row and row[pathKey] == pathValue:
            newRow = copy.deepcopy(row)
            del newRow[pathKey]
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
    return "NaN"

def getKeyFromPath(path : list[dict]):
    nodeClassPathList = []
    for innerPath in path:
        nodePathKey, nodePathValue = list(innerPath.keys())[0], list(innerPath.values())[0]
        nodeClassPathList.append(f"{nodePathKey}:{nodePathValue}")
    return "/".join(nodeClassPathList)

def getNodesFromDataset(dataset : list[dict], classToCheck : str):
    classToCheckValues = getPossibleClassValuesFromDataset(classToCheck, dataset)

    nodes = {"":Node("Root")}
    pathsToCheck = [[]]
    valuesChecked = 0

    startTime = time.time()
    lastLine = ""
    while len(pathsToCheck) > 0:
        print(" "*len(lastLine),end="\r")
        elapsedTime = time.time() - startTime
        lastLine = f"Time Elapsed: {round(elapsedTime,2)}s // Nodes Created: {len(nodes)} // Paths Checked: {valuesChecked} // Percentage Complete: {round((valuesChecked/len(nodes))*100,2)}%"
        print(lastLine,end="\r")
        valuesChecked += 1

        pathToCheck = pathsToCheck.pop(0)
        currentDataset = copy.deepcopy(dataset)

        #Filter Dataset Down
        for path in pathToCheck:
            currentDataset = filterDataset(currentDataset, path)

        nodeClassPath = getKeyFromPath(pathToCheck)
        rootNode = nodes[nodeClassPath]
        
        mainClassCheckEntropy = calculateEntropyFromDataset(classToCheck, currentDataset)
        classesToCheck = [x for x in list(currentDataset[0].keys()) if x != classToCheck]

        bestFittingClass = {"class":"NaN","infoGain":-100.0}
        for dClass in classesToCheck:
            classEntropys = calculateClassValueEntropyFromDataset(dClass, currentDataset, classToCheck, classToCheckValues)
            classInformationGain = calculateClassInformationGainFromDataset(classEntropys, currentDataset, mainClassCheckEntropy)
            
            if classInformationGain > bestFittingClass["infoGain"]:
                bestFittingClass = {"class":dClass,"infoGain":classInformationGain,"entropys":classEntropys}

        rootNode.Class = bestFittingClass["class"]

        classValueLeaves, classValuesToExpand = getValuesLeavesExpandFromBestClass(bestFittingClass)
        for classValueLeaf in classValueLeaves:
            rootNode.addDecision(classValueLeaf["value"], classValueLeaf["result"])
        for classValueToExpand in classValuesToExpand:
            newNode = Node("NaN")
            rootNode.addChild(classValueToExpand, newNode)
            pathToAdd = copy.deepcopy(pathToCheck)
            pathToAdd.append({bestFittingClass["class"]:classValueToExpand})
            nodesKeyPath = getKeyFromPath(pathToAdd)
            nodes[nodesKeyPath] = newNode
            pathsToCheck.append(pathToAdd)

    print(" "*len(lastLine),end="\r")
    elapsedTime = time.time() - startTime
    lastLine = f"Time Elapsed: {round(elapsedTime,2)}s // Nodes Created: {len(nodes)} // Paths Checked: {valuesChecked} // Percentage Complete: {round((valuesChecked/len(nodes))*100,2)}%"
    print(lastLine,end="\r")

    print("\n")
    return nodes

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

def validateDataset(dataset : list[dict], nodes : dict[Node], checkClass : str):
    valid = 0
    for i, entry in enumerate(dataset):
        if entry[checkClass] == getResultOfDatasetEntry(entry, nodes[""]):
            valid += 1

    return valid, len(dataset)

def splitDataset(dataset : list[dict], trainingPercentage : float, testingPercentage : float):
    shuffledDataset = copy.deepcopy(dataset)
    random.shuffle(shuffledDataset)
    trainingDataset = []
    testingDataset = []
    for row in shuffledDataset:
        if len(trainingDataset) < len(shuffledDataset)*trainingPercentage:
            trainingDataset.append(row)
        elif len(testingDataset) < len(shuffledDataset)*testingPercentage:
            testingDataset.append(row)
    return trainingDataset, testingDataset

def getSetAmountFromDataset(dataset : list[dict], checkClass : str, amountOfEach : dict[str, int]):
    #Takes a dictionary of class values and the amount of each to get from the dataset, then returns a new dataset with that amount of each class value.
    classToCheckValues = getPossibleClassValuesFromDataset(checkClass, dataset)
    perValueEntry = {x : [] for x in classToCheckValues}
    for row in dataset:
        perValueEntry[row[checkClass]].append(row)

    newDataset = []
    for value, entries in perValueEntry.items():
        random.shuffle(entries)
        if len(entries) >= amountOfEach[value]:
            newDataset.extend(entries[:amountOfEach[value]])
    return newDataset

# def testDatasetPercentages(dataset : list[dict], checkClass : str, percentages : list[float], runs : int = 1):
#     #Takes a list of percentages and creates a tree based on x% of the dataset, then validates it.

#     #bestTree = {"percentage":0, "nodes":{}}

#     #1.0 -> {"valid":0, "total":0}
#     percentageAverages = {}
#     timeRunsStart = time.time()

#     for run in range(runs):
#         for percentage in percentages:
#             newDataset = copy.deepcopy(dataset)
#             amountOfDataset = round(len(newDataset)*percentage)
#             print(f"({run+1}/{runs}) Testing with {round(len(dataset)*percentage)} items ({round(percentage*100, 2)}% of the dataset).")
#             if amountOfDataset <= 0:
#                 print("Not enough data to test with.")
#                 continue
#             random.shuffle(newDataset)
#             newDataset = newDataset[:amountOfDataset]
#             trainingDataset, testingDataset = splitDataset(newDataset, 0.8, 0.2)
#             nodes = getNodesFromDataset(trainingDataset, checkClass)
#             valid, total = validateDataset(testingDataset, nodes, checkClass)

#             if percentage not in percentageAverages:
#                 percentageAverages[percentage] = {"valid":0, "total":0, "trainingSize":len(trainingDataset),"testingSize":len(testingDataset)}
#             else:
#                 percentageAverages[percentage]["valid"] += valid
#                 percentageAverages[percentage]["total"] += 1
            
#             # if valid/total > bestTree["percentage"]:
#             #     bestTree = {"percentage":valid/total, "nodes":nodes}
            
#             print(f"Valid: {valid}/{total} ({round((valid/total)*100,2)}%).\n")
    
#     print(f"Averages for {runs} runs ({round(time.time()-timeRunsStart,2)}s):")
#     for percentage, data in percentageAverages.items():
#         trainingSize, testingSize = data["trainingSize"], data["testingSize"]
#         averageValid = (data['valid']/data['total'])
#         averagePercentage = round((averageValid/testingSize)*100,2)
#         print(f"Average for {round(percentage*100, 2)}%: Valid: {round(averageValid,2)} / {testingSize} ({averagePercentage}%)")
    # print(f"Best result was {round(bestTree['percentage']*100,2)}% valid, rendered below:")
    # renderNodes(bestTree["nodes"][""], 1, checkClass)

def testDatabaseRatio(dataset : list[dict], checkClass : str, ratios : dict, amount : int = None, runs : int = 1, testFileName : str = None, useOneMinimum : bool = True):
    #Takes a dictionary of ratios and creates a tree based on the ratio of the dataset, then validates it.
    amountOfEach = {k:max(math.floor(v*amount), 1 if useOneMinimum else 0) for k,v in ratios.items()}
    amountOfEachText = ", ".join([f"{k} x {v}" for k,v in amountOfEach.items()])

    bestTree = {"percentage":0, "nodes":{}}

    averageValues = []
    timeRunsStart = time.time()

    for run in range(runs):
        print(f"({run+1}/{runs}) Testing with {amountOfEachText} ({round((amount/len(dataset))*100, 2)}% of the dataset).")
        amountOfEachEntries = getSetAmountFromDataset(dataset, checkClass, amountOfEach)
        nodes = getNodesFromDataset(amountOfEachEntries, checkClass)
        valid, total = validateDataset(dataset, nodes, checkClass)
        averageValues.append(valid)

        if valid/total > bestTree["percentage"]:
            bestTree = {"percentage":valid/total, "nodes":nodes}

        print(f"Valid: {valid}/{total} ({round((valid/total)*100,2)}%)")

    average = sum(averageValues)/len(averageValues)
    print(f"\nAverage for {runs} runs of {amountOfEachText} ({round(time.time()-timeRunsStart,2)}s): Valid: {round(average)}/{len(dataset)} ({round((average/len(dataset))*100,2)}%)")

    print(f"Best result was {round(bestTree['percentage']*100,2)}% valid with {len(bestTree['nodes'])} nodes, rendered below:")
    renderNodes(bestTree["nodes"][""], 1, checkClass)
    if isinstance(testFileName, str):
        with open(f"{testFileName}.data", "wb") as f:
            pickle.dump(bestTree["nodes"], f)

def testDatasetSplitPlot(dataset : list[dict], checkClass : str, splits : list[tuple[float, float]], runs : int = 1):
    percentages = []
    nodeCount = []
    for trainPerc, testPerc in splits:
        for run in range(runs):
            print(f"({run+1}/{runs}) of {trainPerc} / {testPerc} split.")
            trainingDataset, testingDataset = splitDataset(dataset, trainPerc, testPerc)
            nodes = getNodesFromDataset(trainingDataset, checkClass)
            valid, total = validateDataset(testingDataset, nodes, checkClass)
            percentages.append(valid/total)
            nodeCount.append(len(nodes))
    
    plt.xlabel("Node Count")
    plt.ylabel("Validation Percentage")
    plt.scatter(nodeCount, percentages)
    plt.show()



if __name__ == "__main__":
    loadedDataset = extractDatasetFromCSV("courseworkDataset.csv")
    checkClass = list(loadedDataset[0].keys())[-1]
    #, (0.6, 0.4), (0.7, 0.3), (0.8, 0.2), (0.9, 0.1)
    testDatasetSplitPlot(loadedDataset, checkClass, [(0.1, 0.9), (0.2, 0.8), (0.3, 0.7), (0.4, 0.6), (0.5, 0.5)], 10)

    # trainingDataset, testingDataset = splitDataset(loadedDataset, 0.1, 0.9)
    # nodes = getNodesFromDataset(trainingDataset, checkClass)
    # valid, total = validateDataset(testingDataset, nodes, checkClass)
    # print(f"Valid: {valid}/{total} ({round((valid/total)*100,2)}%)")
    # renderNodes(nodes[""], 1, checkClass)

    # with open("Outputs/fullTree.data", "wb") as f:
    #     pickle.dump(nodes, f)

    

    #0.7, 0.2, 0.04, 0.04

    #testDatasetPercentages(loadedDataset, checkClass, [1.0, 0.75, 0.5, 0.25, 0.1, 0.05, 0.01], 50)

    #testDatabaseRatio(dataset=loadedDataset, checkClass=checkClass, ratios={"unacc":0.7, "acc":0.22, "good":0.04, "vgood":0.04}, amount=20, runs=100, testFileName="Outputs/notImportant", useOneMinimum=True)