import math

dataset = [{"shape":"cylinder","color":"orange","volume":25,"sick":"no"},
           {"shape":"cylinder","color":"black","volume":25,"sick":"no"},
           {"shape":"coupe","color":"white","volume":10,"sick":"no"},
           {"shape":"trapezoid","color":"green","volume":15,"sick":"no"},
           {"shape":"coupe","color":"yellow","volume":15,"sick":"no"},
           {"shape":"trapezoid","color":"orange","volume":15,"sick":"yes"},
           {"shape":"coupe","color":"orange","volume":15,"sick":"yes"},
           {"shape":"coupe","color":"orange","volume":10,"sick":"yes"},
           ]

def calculateEntropyFromDataset(parameter : str, dataset : list[dict], clearedValues : dict):
  allowedRows = getAllowedRows(dataset, clearedValues)
  probabilities = {}
  # probabilities = {row[rootParam] : 0 for row in allowedRows}
  paramValues = [row[parameter] for row in allowedRows]
  for x in set(paramValues):
    probabilities[x] = paramValues.count(x)/len(paramValues)
  return -sum([prob * math.log2(prob) for x, prob in probabilities.items()]), len(paramValues)

def getAllowedRows(dataset : list[dict], clearedValues : dict):
  allowedRows = []
  for row in dataset:
    validRow = True
    for rowK, rowV in row.items():
      if rowK in clearedValues and rowV in clearedValues[rowK]:
        validRow = False
        break
    if validRow:
      allowedRows.append(row)
  return allowedRows

def calculateValueEntropys(parameter : str, dataset : list[dict], rootParam : str, clearedValues : dict):
  allowedRows = getAllowedRows(dataset, clearedValues)
  paramValues = set([x[parameter] for x in allowedRows])
  paramProbabilities = {pV : {rD[rootParam] : 0 for rD in allowedRows} for pV in paramValues}
  for row in allowedRows:
    paramVal, rootVal = row[parameter], row[rootParam]
    paramProbabilities[paramVal][rootVal] += 1
  print(paramProbabilities)
  paramEntropys = {}
  entropyValues = []
  for paramVal, results in paramProbabilities.items():
    totalResults = sum([rV for rK, rV in results.items()])
    probabilities = [(rV / totalResults) for rK, rV in results.items()]
    entropy = -sum([prob * math.log2(prob) if prob else 0 for prob in probabilities])
    paramEntropys[paramVal] = {"entropy":entropy,"total":totalResults}
    entropyValues.append(entropy)
  lowestEntropy = min(entropyValues)
  valuesToClear = [pK for pK, pV in paramEntropys.items() if pV["entropy"] == lowestEntropy]
  return paramEntropys, valuesToClear

def calculateInformationGain(valueEntropys : dict[dict], datasetEntropy : float, datasetLength : int):
  return datasetEntropy - sum([(vE["total"]/datasetLength) * vE["entropy"] for vK, vE in valueEntropys.items()])

rootParam = "sick"

usedParams = [rootParam]

clearedValues = {}

datasetKeys = list(dataset[0].keys())

datasetEntropy, datasetLength = calculateEntropyFromDataset(rootParam, dataset, clearedValues)

while (len(usedParams) != len(datasetKeys)) and len(getAllowedRows(dataset, clearedValues)) != 0:

  infoGains = {}

  for value in [x for x in datasetKeys if x not in usedParams]:

    valueEntropy, valuesToClear = calculateValueEntropys(value,dataset,rootParam, clearedValues)
    print(valueEntropy)

    infoGain = calculateInformationGain(valueEntropy, datasetEntropy, datasetLength)

    infoGains[value] = {"gain":infoGain, "valuesToClear":valuesToClear}

  sortedInfoGains = {k: v for k, v in sorted(infoGains.items(), key=lambda item: item[1]["gain"])}
  chosenValue = list(sortedInfoGains.keys())[-1]
  usedParams.append(chosenValue)
  clearedValues[chosenValue] = sortedInfoGains[chosenValue]["valuesToClear"]
  datasetEntropy, datasetLength = calculateEntropyFromDataset(chosenValue, dataset, clearedValues)
  rootParam = chosenValue

  ## 1 bit of entropy
  #print(f"{value} : {infoGain}")