import math
import copy

class Node:
    def __init__(self, paramClass):
        self.children = []
        self.decisions = {}
        self.paramClass = paramClass
    
    def addChild(self, node):
        self.children.append(node)

    def addDecision(self, classValue, result):
        self.decisions [classValue] = result

def calculateEntropyFromDataset(parameter : str, dataset : list[dict]):
  probabilities = {}
  paramValues = [row[parameter] for row in dataset]
  for x in set(paramValues):
    probabilities[x] = paramValues.count(x)/len(paramValues)
  return -sum([prob * math.log2(prob) for x, prob in probabilities.items()]), len(paramValues)

def filterDataset(dataset : list[dict], path : dict):
    pathKey, pathValue = list(path.items())[0]
    newDataset = []
    for row in dataset:
        if pathKey in row and row[pathKey] == pathValue:
            newRow = copy.deepcopy(row)
            del newRow[pathKey]
            newDataset.append(newRow)
    return newDataset

dataset = [{"shape":"cylinder","color":"orange","volume":25,"sick":"no"},
           {"shape":"cylinder","color":"black","volume":25,"sick":"no"},
           {"shape":"coupe","color":"white","volume":10,"sick":"no"},
           {"shape":"trapezoid","color":"green","volume":15,"sick":"no"},
           {"shape":"coupe","color":"yellow","volume":15,"sick":"no"},
           {"shape":"trapezoid","color":"orange","volume":15,"sick":"yes"},
           {"shape":"coupe","color":"orange","volume":15,"sick":"yes"},
           {"shape":"coupe","color":"orange","volume":10,"sick":"yes"},
           ]

classToCheck = "sick"

nodes = {"root":Node("root")}
rootNode = nodes["root"]

#pathsToCheck = [{"node":"shape","path":[{"shape":"cylinder"}]}]
pathsToCheck = [{"node":"root","path":[]}]

while len(pathsToCheck) > 0:
    pathToCheck = pathsToCheck.pop(0)
    currentDataset = copy.deepcopy(dataset)
    currentNode = nodes[pathToCheck["node"]]

    #Filter Dataset Down
    for path in pathToCheck["path"]:
        currentDataset = filterDataset(currentDataset, path)
    
    mainClassCheckEntropy = calculateEntropyFromDataset(classToCheck, currentDataset)
    
    classesToCheck = [x for x in list(currentDataset[0].keys()) if x != classToCheck]

    print(classesToCheck)
    


