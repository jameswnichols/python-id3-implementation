import math
import copy
import pickle
import time
import shutil
import random

TERMINAL_SIZE = shutil.get_terminal_size((80, 20))

class Node:
    def __init__(self, classValue):
        self.children = {}
        self.decisions = {}
        self.Class = classValue
        self.parentClass = ""
    
    def addChild(self, value, node):
        self.children[value] = node

    def addDecision(self, classValue, result):
        self.decisions [classValue] = result

def calculateEntropyFromDataset(parameter : str, dataset : list[dict]):
  probabilities = {}
  paramValues = [row[parameter] for row in dataset]
  for x in list(dict.fromkeys(paramValues)):
    probabilities[x] = paramValues.count(x)/len(paramValues)
  return -sum([prob * math.log2(prob) for x, prob in probabilities.items()])

def calculateClassValueEntropyFromDataset(paramClass : str, dataset : list[dict], checkClass : str, checkClassValues : list[str]):
    paramValues = list(dict.fromkeys([row[paramClass] for row in dataset]))
    paramProb = {pV : {cV : 0 for cV in checkClassValues} for pV in paramValues}
    for row in dataset:
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
    #rootNode = nodes["Root"]

    #pathsToCheck = [{"node":"shape","path":[{"shape":"cylinder"}]}]
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
            classInformationGain = calculateClassInformationGainFromDataset(classEntropys, dataset, mainClassCheckEntropy)
            if classInformationGain > bestFittingClass["infoGain"]:
                bestFittingClass = {"class":dClass,"infoGain":classInformationGain,"entropys":classEntropys}

        rootNode.Class = bestFittingClass["class"]

        #print(f"{pathToCheck} :: {bestFittingClass['class']}")

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

if __name__ == "__main__":
    loadedDataset = extractDatasetFromCSV("courseworkDataset.csv")
    checkClass = list(loadedDataset[0].keys())[-1]

    nodes = getNodesFromDataset(loadedDataset, checkClass)

    with open("nodesOutput.data", "wb") as f:
        pickle.dump(nodes, f)

    renderNodes(nodes[""],1,"quality")

    randomEntry = random.choice(loadedDataset)

    result = getResultOfDatasetEntry(randomEntry, nodes[""])

    print(f"Random Entry: {randomEntry} Result: {result}")

