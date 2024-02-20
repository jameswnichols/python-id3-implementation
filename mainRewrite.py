import math
import copy

class Node:
    def __init__(self, paramClass, parentClassValue):
        self.children = {}
        self.decisions = {}
        self.paramClass = paramClass
        self.parentClassValue = parentClassValue
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
    for value, entropyData in bestFittingClass["entropys"].items():
        if entropyData["entropy"] == 0.0:
            entropyResultValue = [rK for rK, rV in entropyData["results"].items() if rV > 0][0]
            leaves.append({"value":value,"result":entropyResultValue})
        else:
            expand.append(value)
    return leaves, expand

def renderNodes(node : Node, indent : int, checkClassName : str):

    childrenCount = len(node.children)

    endText = "" if childrenCount else "- END"

    spacing = "  " * (indent - 1) + f"If {node.parentClass} = {node.parentClassValue}; check {node.paramClass}" if node.parentClassValue else "↳ "
    print(spacing)
    
    for classValue, result in node.decisions.items():
        print("  " * (indent) + f"↳ If {node.paramClass} = {classValue}; {checkClassName} = {result} {endText}")

    for value, childNode in node.children.items():
        renderNodes(childNode, indent + 1, checkClassName)

def getNodesFromDataset(dataset : list[dict], classToCheck : str):
    #classToCheck = "Play"
    classToCheckValues = getPossibleClassValuesFromDataset(classToCheck, dataset)

    nodes = {"Root":Node("Root","")}
    rootNode = nodes["Root"]

    #pathsToCheck = [{"node":"shape","path":[{"shape":"cylinder"}]}]
    pathsToCheck = [{"node":"Root","path":[]}]
    previousNodes = []

    while len(pathsToCheck) > 0:
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

        nodes[bestFittingClass["class"]] = Node(bestFittingClass["class"], nodePathValue)
        newNode = nodes[bestFittingClass["class"]]
        newNode.parentClass = rootNode.paramClass
        rootNode.addChild(nodePathValue, newNode)

        classValueLeaves, classValuesToExpand = getValuesLeavesExpandFromBestClass(bestFittingClass)
        for classValueLeaf in classValueLeaves:
            newNode.addDecision(classValueLeaf["value"], classValueLeaf["result"])
            

        if bestFittingClass["class"] not in previousNodes:
            
            for classValueToExpand in classValuesToExpand:
                pathToAdd = copy.deepcopy(pathToCheck)
                pathToAdd["path"].append({bestFittingClass["class"]:classValueToExpand})
                pathToAdd["node"] = bestFittingClass["class"]
                pathsToCheck.append(pathToAdd)
            previousNodes.append(bestFittingClass["class"])
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

practicalDataset = [{"shape":"cylinder","color":"orange","volume":25,"sick":"no"},
           {"shape":"cylinder","color":"black","volume":25,"sick":"no"},
           {"shape":"coupe","color":"white","volume":10,"sick":"no"},
           {"shape":"trapezoid","color":"green","volume":15,"sick":"no"},
           {"shape":"coupe","color":"yellow","volume":15,"sick":"no"},
           {"shape":"trapezoid","color":"orange","volume":15,"sick":"yes"},
           {"shape":"coupe","color":"orange","volume":15,"sick":"yes"},
           {"shape":"coupe","color":"orange","volume":10,"sick":"yes"},
           ]

dataset = [
    {"Outlook":"Sunny","Temperature":"Hot","Humidity":"High","Windy":"False","Play":"No"},
    {"Outlook":"Sunny","Temperature":"Hot","Humidity":"High","Windy":"True","Play":"No"},
    {"Outlook":"Overcast","Temperature":"Hot","Humidity":"High","Windy":"False","Play":"Yes"},
    {"Outlook":"Rainy","Temperature":"Mild","Humidity":"High","Windy":"False","Play":"Yes"},
    {"Outlook":"Rainy","Temperature":"Cool","Humidity":"Normal","Windy":"False","Play":"Yes"},
    {"Outlook":"Rainy","Temperature":"Cool","Humidity":"Normal","Windy":"True","Play":"No"},
    {"Outlook":"Overcast","Temperature":"Cool","Humidity":"Normal","Windy":"True","Play":"Yes"},
    {"Outlook":"Sunny","Temperature":"Mild","Humidity":"High","Windy":"False","Play":"No"},
    {"Outlook":"Sunny","Temperature":"Cool","Humidity":"Normal","Windy":"False","Play":"Yes"},
    {"Outlook":"Rainy","Temperature":"Mild","Humidity":"Normal","Windy":"False","Play":"Yes"},
    {"Outlook":"Sunny","Temperature":"Mild","Humidity":"Normal","Windy":"True","Play":"Yes"},
    {"Outlook":"Overcast","Temperature":"Mild","Humidity":"High","Windy":"True","Play":"Yes"},
    {"Outlook":"Overcast","Temperature":"Hot","Humidity":"Normal","Windy":"False","Play":"Yes"},
    {"Outlook":"Rainy","Temperature":"Mild","Humidity":"High","Windy":"True","Play":"No"}
]

courseworkDataset = extractDatasetFromCSV("courseworkDataset.csv")

nodes = getNodesFromDataset(courseworkDataset, "quality")

renderNodes(nodes["Root"],1,"quality")

