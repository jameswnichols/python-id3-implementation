import pandas as pd
from math import log2
import copy
class Node:
    def __init__(self, classValue, targetClass):
        self.children = {}
        self.decisions = {}
        self.Class = classValue
        self.targetClass = targetClass
    
    def addChild(self, value, node):
        self.children[value] = node

    def addDecision(self, classValue, result):
        self.decisions [classValue] = result

    def render(self):
        print(f"Check {self.Class}:")
        self.renderChildNodes(indent=1)

    def renderChildNodes(self, indent : int = 1):
        stringIndent = "  " * indent
        for decision, result in self.decisions.items():
            print(f"{stringIndent}If {self.Class} = {decision}; {self.targetClass} = {result}")
        for value, child in self.children.items():
            print(f"{stringIndent}If {self.Class} = {value}; Check {child.Class}:")
            child.renderChildNodes(indent=indent+1)


def findEntropyOfAttribute(attribute : str, dataset : pd.DataFrame):
    probabilities = dataset[attribute].value_counts() / len(dataset)
    return -sum([prob * log2(prob) if prob else 0 for prob in probabilities.values])

def findEntropyOfValues(attribute : str, dataset : pd.DataFrame, targetClass : str):
    attributeColumn = dataset[attribute]
    uniqueValues = attributeColumn.unique()
    count = {uniqueValue : dataset.where(dataset[attribute] == uniqueValue)[targetClass].dropna(axis=0, inplace=False).value_counts() for uniqueValue in uniqueValues}
    valueEntropy = {uniqueValue : -sum([(val / uniqueCount.sum()) * log2(val / uniqueCount.sum()) if val / uniqueCount.sum() else 0 for val in uniqueCount.values]) for uniqueValue, uniqueCount in count.items()}
    return {uniqueValue : {"entropy":uniqueEntropy, "count":count[uniqueValue]} for uniqueValue, uniqueEntropy in valueEntropy.items()}

def findInformationGain(attributeData : dict[str], targetClassEntropy : float, dataset : pd.DataFrame):
    return targetClassEntropy - sum((cEV["count"].sum() / len(dataset)) * cEV["entropy"] for cEV in list(attributeData.values()))

def getDecisionsAndBranches(attributeEntropy : dict[str]):
    decisions = {}
    branches = []
    for attributeValue, entropyData in attributeEntropy.items():
        if entropyData["entropy"] == 0.0:
            decisions[attributeValue] = entropyData["count"].axes[0][0]
        else:
            branches.append(attributeValue)
    return decisions, branches

def getTreeRootNode(dataset : pd.DataFrame):
    targetClass = list(dataset)[-1]
    rootNode = Node(targetClass, targetClass)
    getChildNode(dataset, rootNode, targetClass)
    return rootNode

def getChildNode(dataset : pd.DataFrame, node : Node, targetClass : str):
    childNodeDataset = copy.deepcopy(dataset)
    targetClassEntropy = findEntropyOfAttribute(targetClass, childNodeDataset)
    attributesToCheck = list(childNodeDataset)
    attributesToCheck.remove(targetClass)
    highestInfoGain = {"attribute":None, "infoGain":-100.0}
    for attribute in attributesToCheck:
        attributeEntropy = findEntropyOfValues(attribute, childNodeDataset, targetClass)
        informationGain = findInformationGain(attributeEntropy, targetClassEntropy, childNodeDataset)
        if informationGain > highestInfoGain["infoGain"]:
            highestInfoGain = {"attribute":attribute, "infoGain":informationGain, "entropy":attributeEntropy}
    node.Class = highestInfoGain["attribute"]
    decisions, branches = getDecisionsAndBranches(highestInfoGain["entropy"])
    for attrValue, mainValue in decisions.items():
        node.addDecision(attrValue, mainValue)
    for branch in branches:
        newNode = Node("NaN", targetClass)
        node.addChild(branch, newNode)
        newNodeDataset = childNodeDataset.mask(childNodeDataset[highestInfoGain["attribute"]] == branch)
        newNodeDataset.drop(highestInfoGain["attribute"], axis=1, inplace=True)
        newNodeDataset.dropna(axis=0, inplace=True)
        getChildNode(newNodeDataset, newNode, targetClass)

if __name__ == "__main__":
    targetClass = "Play"
    loadedDataset = pd.read_csv("tennisDataset.csv")
    rootNode = getTreeRootNode(loadedDataset)
    rootNode.render()