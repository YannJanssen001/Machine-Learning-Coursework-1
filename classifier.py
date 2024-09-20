# classifier.py
# Lin Li/26-dec-2021
#
# Use the skeleton below for the classifier and insert your code here.

import numpy as np

#This is a decision node class, which is used to create the decision tree.
class Node:
    def __init__(self, feature):
        #The feature index on which to split the data
        self.feature = feature
        #Dictionary mapping feature values to child nodes
        self.children = {}

    #Function to add a child node to the current node for a distinct feature value
    def add_child(self, value, node):
        self.children[value] = node
        
#This is the main Classifier class, where the decision tree classification is implemented.
class DecisionTree:
    #Sets the tree, and max depth initially to None
    def __init__(self, max_depth=None):
        #The root of the decision tree
        self.tree = None
        #Max depth of the decision tree
        self.max_depth = max_depth

    #This function calculates the entropy of the target array
    def entropy(self, target): 
        #If target array is empty, return 0 
        if target.size == 0:
            return 0
        entropy = 0
        #Gets the unique values in the target array
        unique_target = np.unique(target) 
        for i in unique_target:
            #Calculates the probability of each target value in the target array
            p = (target[target == i].shape[0] / target.shape[0])
            if p > 0:
                #Calculation for entropy
                entropy -= p * np.log2(p)
        return entropy
    
    #Function to calculate the information gain of a feature in relation to the target
    def information_gain(self, feature, target):
        total_entropy = self.entropy(target)
        #Gets the unique values in the feature array
        unique_features = np.unique(feature)
        weighted_entropy = 0

        #Calculates the weighted entropy for each unique value in the feature array
        for i in unique_features:
            p = (feature[feature == i].shape[0] / feature.shape[0])
            weighted_entropy += p * self.entropy(target[feature == i])    

        #Calculates the information gain and returns it.
        return total_entropy - weighted_entropy
    
    #Method to fit the decision tree to the data
    def fit(self, data, target):
        #Converts the data and target arrays to numpy arrays, and then creates the decision tree
        data = np.array(data) 
        target = np.array(target)
        self.tree = self.create_tree(data, target, 0)

    #Method to create the decision tree
    def create_tree(self, data, target, depth):
        #If the max depth is reached, return the average of the target array
        if depth == self.max_depth:
            return np.mean(target)
        #If the target array has only one unique value, return that value
        if np.all(target == target[0]):
            return target[0]
        best_gain = 0
        best_feature = None
        #Iterates through each feature in the data array to find the feature with the highest information gain
        for i in range(data.shape[1]):
            gain = self.information_gain(data[:, i], target)
            if gain > best_gain:
                best_gain = gain
                best_feature = i
        if best_feature is None:
            return np.mean(target)
        #Creates a new node with the best feature
        tree = Node(best_feature)

        #Iterates through each unique value in the best feature and creates a child node for each value
        unique_best_feat = np.unique(data[:, best_feature])
        for value in unique_best_feat:
            #Creates a subset of data and target for each unique value in the best feature
            subset_data = data[data[:, best_feature] == value]
            subset_target = target[data[:, best_feature] == value]
            #Creates a child node for each unique value in the best feature
            child = self.create_tree(subset_data, subset_target, depth + 1)
            tree.add_child(value, child)
        return tree

    #Predicts the output for a given input
    def predict(self, data):
        #Converts the data into to a numpy array, and if the data array is 2D,
        #returns an array of predictions for each row in the array.
        data = np.array(data)
        if data.ndim == 2:
            return np.array([self.predict(row) for row in data])
        #Start at the root of the tree
        node_or_value = self.tree

        #Keep looping while still at a decision node
        while isinstance(node_or_value, Node):
            #Based on the data's feature value, move to the appropriate child node
            node_or_value = node_or_value.children.get(data[node_or_value.feature], None)

            #If there is no child node, stop the loop
            if node_or_value is None:
                break
        #And return the output
        return node_or_value

#This function converts the target array into a one-hot encoded array
def one_hot_encode(targets):
    one_hot = np.zeros((len(targets), len(np.unique(targets))))
    #Iterates through each target value and creates a one-hot encoded array
    for i in range(len(targets)):
        one_hot[i, int(targets[i])] = 1
        #returns the one hot encoded array
    return one_hot

#This is the gradient boosting classifier class, using the decision tree class
class Classifier:
    #Initializes the classifier with the max depth, learning rate, and number of trees
    def __init__(self,max_depth=4, learning_rate=0.2, num_trees=10):
        self.trees = []
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.num_trees = num_trees

    #Resets the classifier
    def reset(self):
        self.trees = []

    #Fits the classifier to the data and target
    def fit(self, data, target):
        #Converts the data into a numpy array and the target into a one-hot encoded array
        data = np.array(data)
        one_hot_target = one_hot_encode(target)
        #Initializes the predictions to an array of zeros
        preds = np.zeros_like(one_hot_target)
        #Fits the classifier to the data and target by iteratively creating decision trees
        for i in range(self.num_trees):
            trees = []
            #Calculates the negative gradients
            neg_grads = one_hot_target - preds
            #Iterates through each target column and creates a decision tree for each column
            for j in range(one_hot_target.shape[1]):
                tree = DecisionTree(self.max_depth)
                tree.fit(data, neg_grads[:, j])
                trees.append(tree)
                #Updates the predictions with the new decision tree
                preds[:, j] += self.learning_rate * tree.predict(data)
            self.trees.append(trees)

    #Predicts the output for a given input data
    def predict(self, data, legal=None):
        #Converts the data into a numpy array
        data = np.array(data)
        #If the data array is 2D, returns an array of predictions for each row in the array
        preds = np.zeros(len(self.trees[0]))
        #iterates through each tree in the classifier and makes a prediction 
        #based on the gradient boosted trees
        for trees in self.trees:
            for i in range(len(trees)):
                preds[i] += self.learning_rate * trees[i].predict(data)
        #Returns the index of the maximum value in the predictions array
        return np.argmax(preds)