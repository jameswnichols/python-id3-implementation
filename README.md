# Python ID3 Implementation
A Pure-Python implementation of the ID3 algorithm with reasonable optimisations made, written for my first-year coursework. Can generate a tree based on ~1.7k entries in 20ms and a tree based of ~370k entries in 90 minutes.

### Usage:
**An example is in `main.py`**

Start by importing the decision tree using:
```python
from id3 import DecisionTree
```

Create a new tree class using the code below, optionally passing in a csv filepath and if it has headers:
```python
tree = DecisionTree(csvfilePath="tennisDataset.csv", csvHasHeaders=True)
#OR
tree = DecisionTree()
tree.extractDatasetFromCSV(filePath="tennisDataset.csv", fileHasHeaders=True)
```

Generate the tree using the code below, optionally passing in a training percentage. If one isn't given, the whole dataset will be used:
```python
tree.train()
```

Visualise the tree using:
```python
tree.render()
```

The result of visualising `tennisDataset.csv` is:
```
Check Outlook:
  If Outlook = Overcast; Play = Yes
  If Outlook = Sunny; Check Humidity:
    If Humidity = Normal; Play = Yes
    If Humidity = High; Play = No
  If Outlook = Rainy; Check Windy:
    If Windy = True; Play = No
    If Windy = False; Play = Yes
```

Validate the dataset, returning the amount it got correct and the total rows of testing data:
```python
valid, total = tree.test()
```

To get the best tree out of `n` runs use:
```python
bestTreeResults = tree.findBestTree(trainingSetPercentage=0.4, minimumPercentage=0.7, runs=10)
```
Minimum percentage is used when testing the tree, where if the result is lower, the tree is discarded.


`findBestTree` returns the class `BestTreeResults` with the following attributes:
```python
BestTreeResults.rootNode
BestTreeResults.totalNodes
BestTreeResults.percentage
```

`rootNode` Can be used with:
```python
tree.renderNode(bestTreeResults.rootNode)
```
To render the best tree, as shown above.

`totalNodes` and `percentage` Are the best trees node-count and accuracy respectively.

### Requirements:
Was written in python 3.10 so should work on 3.10+. Any versions before that your mileage may vary.
