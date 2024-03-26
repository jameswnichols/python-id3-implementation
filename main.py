from id3 import DecisionTree

if __name__ == "__main__":
    decisionTree = DecisionTree("courseworkDataset.csv", True)
    decisionTree.getNodesFromDataset(trainingPercentage=0.4)
    decisionTree.validateDataset()
    decisionTree.renderTree()
    #bestTreeResults = decisionTree.testFindBestTree(trainingSetPercentage=0.4, minimumPercentage=0.7, runs=10)