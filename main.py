import math
import copy
import pickle
import time
import shutil

TERMINAL_SIZE = shutil.get_terminal_size((80, 20))

class Node:
    def __init__(self, paramClass):
        self.children = {}
        self.decisions = {}
        self.paramClass = paramClass
        self.parentClassValue = []
        self.parentClass = ""
    
    def addChild(self, value, node):
        self.children[value] = node

    def addParentClassValue(self, value):
        self.parentClassValue.append(value)
        #self.parentClassValue = value# = #list(dict.fromkeys(self.parentClassValue))

    def addDecision(self, classValue, result):
        self.decisions [classValue] = [result]
        # if classValue not in self.decisions:
        #     self.decisions [classValue] = [result]
        # else:
        #     self.decisions [classValue].append(result)
        # self.decisions[classValue] = list(dict.fromkeys(self.decisions[classValue]))

class ClassValueCheck:
    def __init__(self):
        self.classValues = {}
    
    def canUseClassValue(self, pClass : str, pValue : str):
        if pClass not in self.classValues:
            return True
        
        if pValue not in self.classValues[pClass]:
            return True
        
        return False
    
    def addClassValue(self, pClass : str, pValue : str):
        if pClass not in self.classValues:
            self.classValues[pClass] = [pValue]
        else:
            self.classValues[pClass].append(pValue)

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

def renderNodes(node : Node, indent : int, checkClassName : str):
    spacing = "  " * (indent - 1) + f"If {node.parentClass} = {node.parentClassValue}; check " if node.parentClassValue else "↳ "
    print(spacing + node.paramClass)
    for classValue, result in node.decisions.items():
        print("  " * (indent) + f"↳ If {node.paramClass} = {classValue}; {checkClassName} = {result} - END")
    for _, childNode in node.children.items():
        renderNodes(childNode, indent + 1, checkClassName)

def getNodesFromDataset(dataset : list[dict], classToCheck : str):
    #classToCheck = "Play"
    classToCheckValues = getPossibleClassValuesFromDataset(classToCheck, dataset)

    nodes = {"Root":Node("Root")}
    rootNode = nodes["Root"]
    classValueCheck = ClassValueCheck()

    #pathsToCheck = [{"node":"shape","path":[{"shape":"cylinder"}]}]
    pathsToCheck = [{"node":"Root","path":[]}]
    valuesChecked = 0

    startTime = time.time()
    lastLine = ""

    while len(pathsToCheck) > 0:
        #print(" "*len(lastLine),end="\r")
        elapsedTime = time.time() - startTime
        lastLine = f"Time Elapsed: {round(elapsedTime,2)}s // Nodes Created: {len(nodes)} // Paths Checked: {valuesChecked}"
        #print(lastLine,end="\r")
        valuesChecked += 1
        pathToCheck = pathsToCheck.pop(0)

        currentDataset = copy.deepcopy(dataset)
        rootNode = nodes[pathToCheck["node"]]

        nodePathValue = ""

        #Filter Dataset Down
        for path in pathToCheck["path"]:
            currentDataset = filterDataset(currentDataset, path)
            nodePathValue = list(path.values())[0]
        
        mainClassCheckEntropy = calculateEntropyFromDataset(classToCheck, currentDataset)
        classesToCheck = [x for x in list(currentDataset[0].keys()) if x != classToCheck]

        bestFittingClass = {"class":"NaN","infoGain":-100.0}
        for dClass in classesToCheck:
            
            classEntropys = calculateClassValueEntropyFromDataset(dClass, currentDataset, classToCheck, classToCheckValues)
            classInformationGain = calculateClassInformationGainFromDataset(classEntropys, dataset, mainClassCheckEntropy)
            if classInformationGain > bestFittingClass["infoGain"]:
                bestFittingClass = {"class":dClass,"infoGain":classInformationGain,"entropys":classEntropys}

        print(f"{pathToCheck} :: {bestFittingClass['class']}")

        if bestFittingClass["class"] not in nodes:
            nodes[bestFittingClass["class"]] = Node(bestFittingClass["class"])
            newNode = nodes[bestFittingClass["class"]]
            newNode.addParentClassValue(nodePathValue)
            classValueCheck.addClassValue(rootNode.paramClass, nodePathValue)
            newNode.parentClass = rootNode.paramClass
            rootNode.addChild(nodePathValue, newNode)
        else:
            newNode = nodes[bestFittingClass["class"]]
            if classValueCheck.canUseClassValue(rootNode.paramClass, nodePathValue) and rootNode.paramClass == newNode.parentClass:
                classValueCheck.addClassValue(rootNode.paramClass, nodePathValue)
                newNode.addParentClassValue(nodePathValue)

        classValueLeaves, classValuesToExpand = getValuesLeavesExpandFromBestClass(bestFittingClass)
        for classValueLeaf in classValueLeaves:
            if classValueCheck.canUseClassValue(classValueLeaf["value"], classValueLeaf["result"]):
                newNode.addDecision(classValueLeaf["value"], classValueLeaf["result"])
                classValueCheck.addClassValue(rootNode.paramClass, classValueLeaf["result"])
            
        for classValueToExpand in classValuesToExpand:
            pathToAdd = copy.deepcopy(pathToCheck)
            pathToAdd["path"].append({bestFittingClass["class"]:classValueToExpand})
            pathToAdd["node"] = bestFittingClass["class"]
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

    renderNodes(nodes["Root"],1,"quality")

