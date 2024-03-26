# Python ID3 Implementation
A Pure-Python implementation of the ID3 algorithm with reasonable optimisations made. Can generate a tree based on ~1.7k entries in 20ms and a tree based of ~370k entries in 90 minutes.

### Usage:
**An example is in `main.py`**.
Start by importing the decision tree using:
```python
from id3 import DecisionTree
```

Create a new tree class using:
```python
tree = DecisionTree()
```
Optionally passing in a csv filepath and if it has headers.

Generate the tree using `tree.train()`, optionally passing in a training percentage. If one isn't given, the whole dataset will be used.
Visualise the tree using `tree.render()`.
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


### Requirements:
None.
