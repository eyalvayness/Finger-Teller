from PIL import Image
from IPython.display import display
import numpy as np
import matplotlib.colors as mcolors
from PIL import Image
import matplotlib.pyplot as plt

def extractFromDictionnary(imagesDict):
    allFeaturesList = []
    allTargetList = []

    for i in range(len(imagesDict)):
        features, target = extractFeatures(imagesDict[i])
        allFeaturesList.append(features)
        allTargetList.append(target)

    X = np.vstack(allFeaturesList)
    Y = np.hstack(allTargetList)
    
    return X, Y

def extractFeatures(image):
    boolImage = image['bool']

    featuresList = []
    features = []
    distance = 4
    resizedXMax = len(image['bool']) - distance*2
    resizedYMax = len(image['bool'][0]) - distance*2
    #target = np.zeros(resizedXMax*resizedYMax, dtype='int')
    target = []
    counter = 0

    for i in range(resizedXMax):
        for j in range(resizedYMax):
            box = image['boxs'][counter]

            boxVec = box.ravel()
            featuresList.append(boxVec)
            if sum(boxVec == 0) < len(boxVec):
                if (image['actual'][i, j] == True):
                    target.append(1)
                else:
                    target.append(0)
            else:
                target.append(0)
    
            counter += 1
            
    features = np.vstack(featuresList)
    
    return features, target
#--------------------------------------------------------------------------------------------------------------------#
def visualiseRule(indices):
    vec = np.zeros(81)
    for index in indices:
        if index > 0:
            vec[index] = 1
        elif index < 0:
            vec[abs(index)] = -1
            
    box = vec.reshape(9, 9)
    return box

#--------------------------------------------------------------------------------------------------------------------#
def createRuleFromIndices(indices, rule):
    if(len(indices) == 0):
        return rule
    
    firstIndex = indices[0]
    if firstIndex > 0:
        newRule = lambda x: np.logical_and(x[firstIndex] == True, rule(x) == True)
    elif firstIndex < 0:
        newRule = lambda x: np.logical_and(x[abs(firstIndex)] == False, rule(x) == True)
        
    lastRule = createRuleFromIndices(indices[1:], newRule)
    
    return lastRule

def sortFromSimpleRule(rule, esemble, X):
    leftIndices = []
    rightIndices = []
    for index in esemble:
        vec = X[index]
        if(rule(vec) == True):
            rightIndices.append(index)
        else:
            leftIndices.append(index)
    
    return leftIndices, rightIndices

def computeRootEntropy(Y):
    actual = sum(Y == 1)
    if (actual == 0 or actual == len(Y)):
        return 0
    actualRatio = actual/len(Y)
    
    rootEntropy = actualRatio*np.log2(1/actualRatio)+(1-actualRatio)*np.log2(1/(1-actualRatio))
    return rootEntropy

def computeAnyNodeEntropy(indices, Y, position):
    if (len(indices) == 0):
        return 0, 0, 0
        
    actual = sum(Y[indices] == 1)        
    actualRatio = actual/len(indices)
    
    #Perfect/pure sorting
    if (actualRatio == 1 or actualRatio == 0):
        return 0, actual, len(indices)
    
    entropy = actualRatio*np.log2(1/actualRatio)+(1-actualRatio)*np.log2(1/(1-actualRatio))
    return entropy, actual, len(indices)

def computeInformationGain(index, esemble, X, Y, parentEntropy, initialRule):
    newRule = lambda x: np.logical_and(x[index] == True, initialRule(x) == True)
        
    leftIndices, rightIndices = sortFromSimpleRule(newRule, esemble, X)
    leftEntropy, leftOk, leftAll = computeAnyNodeEntropy(leftIndices, Y, 'left')
    rightEntropy, rightOk, rightAll = computeAnyNodeEntropy(rightIndices, Y, 'right')
    
    leftDistrib = [leftAll - leftOk, leftOk]
    rightDistrib = [rightAll - rightOk, rightOk]
    
    leftRatio = len(leftIndices)/len(esemble)
    rightRatio = len(rightIndices)/len(esemble)
    
    currentInformationGain = parentEntropy - (leftRatio*leftEntropy + rightRatio*rightEntropy)
    distrib = [leftDistrib, rightDistrib]
    entropy = [leftEntropy, rightEntropy]
    indices = [leftIndices, rightIndices]
    return currentInformationGain, distrib, entropy, indices

def addChildNode(parentNode, depth, Y, endNodes):
    if (depth <= 0 or parentNode.distribution[0] == 0 or parentNode.distribution[1] == 0):
        if(len(parentNode.indices) > 1):
            parentNode.indices = parentNode.indices[1:]
        parentNode.rule = createRuleFromIndices(parentNode.indices.copy(), lambda x: (x.any() == x.any()))
        endNodes.append(parentNode)
        return endNodes.copy()
    
    allInformationGain = np.zeros(81)
    distribs = []
    entropies = []
    esembleIndices = []
    maxIndex = 0
    parentEsemble = parentNode.esemble
    parentConvolutions = parentNode.convolutions
    parentRule = parentNode.rule
    parentEntropy = parentNode.entropy
    
    parentIndices = parentNode.indices.copy()
    parentIndex = parentNode.indices[len(parentIndices) - 1]
    
    #parentNode.show()
    #print('--------------------------------')
    #print('depth : ' + str(depth))
    #print(parentIndices)
    #print('length : ' + str(len(parentIndices)))
    #print(parentIndex)
    #print('--------------------------------')
    
    if(parentIndex < 0):
        childrenRule =  lambda x: np.logical_and(parentNode.rule(x) == True, x[abs(parentIndex)] == False)
    elif(parentIndex > 0):
        childrenRule = lambda x: np.logical_and(parentNode.rule(x) == True, x[parentIndex] == True)
    
    for i in range(81):
        allInformationGain[i], distrib, entropy, sortedIndex = \
            computeInformationGain(i, parentEsemble, parentConvolutions, Y, parentEntropy, childrenRule)
        distribs.append(distrib)
        entropies.append(entropy)
        esembleIndices.append(sortedIndex)
        if(allInformationGain[i] > allInformationGain[maxIndex]):
            maxIndex = i
            
    leftChildIndices = parentIndices + [-maxIndex]
    rightChildIndices = parentIndices + [maxIndex]
    
    #print(leftChildIndices)
    #print('length : ' + str(len(leftChildIndices)))
    #print(maxIndex)
    #print('--------------------------------')
    #print(rightChildIndices)
    #print('length : ' + str(len(rightChildIndices)))
    #print('--------------------------------')
    
    leftEsemble = esembleIndices[maxIndex][0]
    rightEsemble = esembleIndices[maxIndex][1]
    
    X = parentNode.convolutions.copy()
    
    leftChild =  Node(childrenRule, leftChildIndices, distribs[maxIndex][0],\
                      entropies[maxIndex][0], leftEsemble, X, None, None)
    rightChild = Node(childrenRule, rightChildIndices, distribs[maxIndex][1],\
                      entropies[maxIndex][1],rightEsemble, X, None, None)
    
    #leftChild.show()    
    #print('--------------------------------')
    #rightChild.show()    
    #print('--------------------------------')
    
    parentNode.leftChild = leftChild
    parentNode.rightChild = rightChild
    
    endNodes = addChildNode(leftChild , depth - 1, Y, endNodes.copy())
    endNodes = addChildNode(rightChild, depth - 1, Y, endNodes.copy())
    
    return endNodes.copy()
    
def computeRootIndex(esemble, X, Y, rootNode):
    allInformationGain = np.zeros(81)
    currentMaxIndex = 0
    
    for i in range(81):
        allInformationGain[i] = computeInformationGain(i, esemble, X, Y, rootNode.entropy, rootNode.rule)[0]
        if(allInformationGain[i] > allInformationGain[currentMaxIndex]):
            currentMaxIndex = i
            
    #RULE = lambda x: (X[currentMaxIndex] == 1)
    return currentMaxIndex

#def createClassifierTree(X, Y, depth):
#    endNodes = []
#    
#    rootEsemble = []
#    for i in range(len(X)):
#        rootEsemble.append(i)
#    
#    rootEntropy = computeAnyNodeEntropy(rootEsemble, Y, 'root')[0]
#    
#    rootRule = lambda x: (x.any() == x.any())
#    #print('The root entropy is : ' + str(rootEntropy))
#    root = Node(rootRule, None, [sum(Y == False), sum(Y == True)], rootEntropy, rootEsemble, X, None, None)
#    root.indices = [computeRootIndex(rootEsemble, X, Y, root)]
#    #root.indices = []
#    #root.indices.append(computeRootIndex(X, Y, root))
#    #root.show()
#    
#    if (depth >= 2):
#        endNodes = addChildNode(root, depth, Y, endNodes)
#    elif (depth <= 1):
#        root.rule = lambda x: (x[root.indices[0]] == True)
#        endNodes.append(root)
#    
#    decisionTree = Tree(root, endNodes)
#    
#    return decisionTree 
       
class Node:
    def __init__(self, rule, indices, distribution, entropy, esemble, convolutions, leftChild, rightChild):
        self.rule = rule
        self.indices = indices
        self.esemble = esemble
        self.convolutions = convolutions
        self.distribution = distribution
        self.entropy = entropy
        self.leftChild = leftChild
        self.rightChild = rightChild
    
    def changeLeftChild(self, leftChild):
        self.leftChild = leftChild
        
    def changeLeftChild(self, leftChild):
        self.leftChild = leftChild
        
    def show(self):
        print('rule : ' + str(self.rule))
        print('indices : ' + str(self.indices[:]))
        print('entropy : ' + str(round(self.entropy*1000)/1000))
        print('distribution : ' + str(self.distribution))
        ratio = self.distribution[1]*100/float(self.distribution[0] + self.distribution[1])
        print('chance of finger : ' + str(float(format(ratio, '.2f')))+'%')
        print('--------------------------------')
        
class Tree:
    def __init__(self, name):
        self.name = name
        
    def fit(self, X, Y, depth):
        endNodes = []
        
        rootEsemble = range(len(X))
        rootEntropy = computeAnyNodeEntropy(rootEsemble, Y, 'root')[0]
        
        rootRule = lambda x: (x.any() == x.any())
        #print('The root entropy is : ' + str(rootEntropy))
        root = Node(rootRule, None, [sum(Y == False), sum(Y == True)], rootEntropy, rootEsemble, X, None, None)
        root.indices = [computeRootIndex(rootEsemble, X, Y, root)]
        #root.indices = []
        #root.indices.append(computeRootIndex(X, Y, root))
        #root.show()
        
        if (depth >= 2):
            endNodes = addChildNode(root, depth, Y, endNodes)
        elif (depth <= 1):
            root.rule = lambda x: (x[root.indices[0]] == True)
            endNodes.append(root)
        
        self.root = root
        self.endNodes = endNodes
        self.optimalRules = self.extractOptimalRules()
        
    def show(self):
        for node in self.endNodes:
            node.show()
          
        visualRules = []
        if(isinstance(self.optimalRules, str) == True):
            print(self.optimalRules)
            return False
        else:
            for index in range(len(self.optimalRules)):
                visualRules.append([visualiseRule(self.optimalRules[index][1]), self.optimalRules[index][1]])
            return visualRules
        
    def extractOptimalRules(self):
        optiRules = []
        for node in self.endNodes:
            if(node.distribution[0] < node.distribution[1]):
                optiRules.append([node.rule, node.indices, node.distribution])
                
        if(len(optiRules) <= 0):
            return 'There is no optimal rules in this tree'    
        elif(len(optiRules) > 0):
            return optiRules
    
    def predictFromOptimalRules(self, X):
        yHat = np.zeros(len(X), dtype='uint8')
        isOK = False
        
        for index in range(len(X)):
            isOK = False
            convolution = X[index]
            if(sum(convolution==0) == len(convolution)):
                yHat[index] = 0
            else:
                for rule in self.optimalRules:
                    if isOK == False:
                        if(rule[0](convolution) == True):
                            yHat[index] = 1
                            isOK = True
        return yHat
    
    def predictFromOutsiderRules(self, X, rules):
        yHat = np.zeros(len(X), dtype='uint8')
        isOK = False
        
        for index in range(len(X)):
            isOK = False
            convolution = X[index]
            if(sum(convolution==0) == len(convolution)):
                yHat[index] = 0
            else:
                for rule in rules:
                    if isOK == False:
                        if(rule[0](convolution) == True):
                            yHat[index] = 1
                            isOK = True
        return yHat