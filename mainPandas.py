import pandas as pd
from math import log2
import copy
class Node:
    def __init__(self, classValue):
        self.children = {}
        self.decisions = {}
        self.Class = classValue
    
    def addChild(self, value, node):
        self.children[value] = node

    def addDecision(self, classValue, result):
        self.decisions [classValue] = result

def findEntropyOfAttribute(attribute : str, dataset : pd.DataFrame):
    probabilities = dataset[attribute].value_counts() / len(dataset)
    return -sum([prob * log2(prob) if prob else 0 for prob in probabilities.values])

def findEntropyOfValues(attribute : str, dataset : pd.DataFrame, targetClass : str):
    attributeColumn = dataset[attribute]
    uniqueValues = attributeColumn.unique()
    count = {uniqueValue : dataset.where(dataset[attribute] == uniqueValue)[targetClass].value_counts() for uniqueValue in uniqueValues}
    valueEntropy = {uniqueValue : -sum([(val / uniqueCount.sum()) * log2(val / uniqueCount.sum()) if val / uniqueCount.sum() else 0 for val in uniqueCount.values]) for uniqueValue, uniqueCount in count.items()}
    return {uniqueValue : {"entropy":uniqueEntropy, "count":count[uniqueValue]} for uniqueValue, uniqueEntropy in valueEntropy.items()}

def findInformationGain(attributeData : dict[str], targetClassEntropy : float, dataset : pd.DataFrame):
    return targetClassEntropy - sum((cEV["count"].sum() / len(dataset)) * cEV["entropy"] for cEV in list(attributeData.values()))

def getTreeRootNode(dataset : pd.DataFrame):
    targetClass = list(dataset)[-1]
    rootNode = Node(targetClass)
    getChildNode(dataset, rootNode, targetClass)

def getChildNode(dataset : pd.DataFrame, node : Node, targetClass : str):
    childNodeDataset = copy.deepcopy(dataset)
    targetClassEntropy = findEntropyOfAttribute(targetClass, childNodeDataset)
    attributesToCheck = list(childNodeDataset).remove(targetClass)
    for attribute in attributesToCheck:
        attributeEntropy = findEntropyOfValues(attribute, childNodeDataset, targetClass)
        informationGain = findInformationGain(attributeEntropy, targetClassEntropy, childNodeDataset)



if __name__ == "__main__":
    targetClass = "Play"
    loadedDataset = pd.read_csv("tennisDataset.csv")
    getTreeRootNode(loadedDataset)