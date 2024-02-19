import math
import copy

class Node:
    def __init__(self, paramClass):
        self.children = []
        self.decisions = {}
        self.paramClass = paramClass
    
    def addChild(self,  node):
        self.children.append(node)

    def addDecision(self, classValue, result):
        self.decisions [classValue] = result

def calculateEntropyFromDataset(parameter : str, dataset : list[dict]):
  probabilities = {}
  paramValues = [row[parameter] for row in dataset]
  for x in set(paramValues):
    probabilities[x] = paramValues.count(x)/len(paramValues)
  return -sum([prob * math.log2(prob) for x, prob in probabilities.items()])

def calculateClassValueEntropyFromDataset(paramClass : str, dataset : list[dict], checkClass : str, checkClassValues : list[str]):
    paramValues = list(set([row[paramClass] for row in dataset]))
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
    possibleClasses = set()
    for row in dataset:
        possibleClasses.add(row[paramClass])
    return list(possibleClasses)

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
    {"outlook":"Sunny","temperature":"Hot","humidity":"High","windy":"False","play":"No"},
    {"outlook":"Sunny","temperature":"Hot","humidity":"High","windy":"True","play":"No"},
    {"outlook":"Overcast","temperature":"Hot","humidity":"High","windy":"False","play":"Yes"},
    {"outlook":"Rainy","temperature":"Mild","humidity":"High","windy":"False","play":"Yes"},
    {"outlook":"Rainy","temperature":"Cool","humidity":"Normal","windy":"False","play":"Yes"},
    {"outlook":"Rainy","temperature":"Cool","humidity":"Normal","windy":"True","play":"No"},
    {"outlook":"Overcast","temperature":"Cool","humidity":"Normal","windy":"True","play":"Yes"},
    {"outlook":"Sunny","temperature":"Mild","humidity":"High","windy":"False","play":"No"},
    {"outlook":"Sunny","temperature":"Cool","humidity":"Normal","windy":"False","play":"Yes"},
    {"outlook":"Rainy","temperature":"Mild","humidity":"Normal","windy":"False","play":"Yes"},
    {"outlook":"Sunny","temperature":"Mild","humidity":"Normal","windy":"True","play":"Yes"},
    {"outlook":"Overcast","temperature":"Mild","humidity":"High","windy":"True","play":"Yes"},
    {"outlook":"Overcast","temperature":"Hot","humidity":"Normal","windy":"False","play":"Yes"},
    {"outlook":"Rainy","temperature":"Mild","humidity":"High","windy":"True","play":"No"}
]

# for row in dataset:
#     print(f'{row["color"]};{row["volume"]};{row["sick"]};{row["shape"]}')

classToCheck = "play"
classToCheckValues = getPossibleClassValuesFromDataset(classToCheck, dataset)

nodes = {"root":Node("root")}
rootNode = nodes["root"]

#pathsToCheck = [{"node":"shape","path":[{"shape":"cylinder"}]}]
pathsToCheck = [{"node":"root","path":[]}]
previousNodes = []

while len(pathsToCheck) > 0:
    pathToCheck = pathsToCheck.pop(0)

    currentDataset = copy.deepcopy(dataset)
    currentNode = nodes[pathToCheck["node"]]

    canContinue = True

    #Filter Dataset Down
    for path in pathToCheck["path"]:

        currentDataset = filterDataset(currentDataset, path)
    
    mainClassCheckEntropy = calculateEntropyFromDataset(classToCheck, currentDataset)
    classesToCheck = [x for x in list(currentDataset[0].keys()) if x != classToCheck]

    bestFittingClass = {"class":"NaN","infoGain":-100.0}
    for dClass in classesToCheck:
        
        classEntropys = calculateClassValueEntropyFromDataset(dClass, currentDataset, classToCheck, classToCheckValues)
        classInformationGain = calculateClassInformationGainFromDataset(classEntropys, dataset, mainClassCheckEntropy)
        if classInformationGain > bestFittingClass["infoGain"]:
            bestFittingClass = {"class":dClass,"infoGain":classInformationGain,"entropys":classEntropys}

    nodes[bestFittingClass["class"]] = Node(bestFittingClass["class"])
    newNode = nodes[bestFittingClass["class"]]
    rootNode.addChild(newNode)

    if bestFittingClass["class"] not in previousNodes:
        print(bestFittingClass["class"])
        classValueLeaves, classValuesToExpand = getValuesLeavesExpandFromBestClass(bestFittingClass)
        for classValueLeaf in classValueLeaves:
            print(classValueLeaf)
            newNode.addDecision(classValueLeaf["value"], classValueLeaf["result"])
        
        for classValueToExpand in classValuesToExpand:
            pathToAdd = copy.deepcopy(pathToCheck)
            pathToAdd["path"].append({bestFittingClass["class"]:classValueToExpand})
            pathToAdd["node"] = bestFittingClass["class"]
            pathsToCheck.append(pathToAdd)

        rootNode = newNode

        previousNodes.append(bestFittingClass["class"])

for node, nodeClass in nodes.items():
    print(nodeClass.children)

