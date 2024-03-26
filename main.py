from tree import Tree

if __name__ == "__main__":
    decisionTree = Tree("courseworkDataset.csv", True)
    decisionTree.getNodesFromDataset(trainingPercentage=0.4)
    decisionTree.validateDataset()
    decisionTree.renderTree()
    #bestTreeResults = decisionTree.testFindBestTree(trainingSetPercentage=0.4, minimumPercentage=0.7, runs=10)