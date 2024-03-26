from id3 import DecisionTree

if __name__ == "__main__":

    #Initialise the decision tree.
    decisionTree = DecisionTree(csvfilePath="courseworkDataset.csv", csvHasHeaders=True)
    #OR
    decisionTree = DecisionTree()
    decisionTree.extractDatasetFromCSV(filePath="courseworkDataset.csv", fileHasHeaders=True)

    #Get the nodes from the dataset, with an optional training percentage.
    decisionTree.train(trainingPercentage=0.4)

    #Render the tree.
    decisionTree.render()

    #Validate the dataset with it's testing dataset.
    valid, total = decisionTree.test()
    print(f"Accuracy was {round(valid/total*100,2)}%")

    #Can find the best tree out of n Runs.
    #bestTreeResults = decisionTree.testFindBestTree(trainingSetPercentage=0.4, minimumPercentage=0.7, runs=10)