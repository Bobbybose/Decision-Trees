# Author: Bobby Bose
# Assignment 1: Decision Trees


# Imports
import numpy as np
import math
import pandas as pd
pd.options.display.max_rows = 1000
import matplotlib.pyplot as plt

# Global variables
MAX_DEPTH = 3
NUM_BINS = 4
SYNTHETIC_CLASS_LABEL = "class"
POKEMON_CLASS_LABEL = "Legendary"


# Description: Nodes (leaves) of the tree are stored in Node class
class Node:
    
    def __init__(self, value = "Root"):
        self.attribute = ""
        self.value = value
        self.child_nodes = []
    
    def __str__(self):
        return "Splitting Attribute: " + str(self.attribute) + "    Value: " + str(self.value)
# Node


# Description: Main function of the program. Just calls functions for Parts 1-3 of Assignment
# Arguments: None
# Returns: None
def main():   

    # Classifying synthetic data and visualizing classifiers
    synthetic_data()

    print("\n-----------------------------------------------------------------------------------\n")

    # Classifying pokemon data
    pokemon_data()
# main()


# Description: Train and test the Decision Trees for the synthetic data
# Arguments: None
# Returns: None
def synthetic_data():

    # PART 1 of Assignment ---------------------------------------------------------------------------------------------------

    # Reading in synthetic data
    synthetic_data_df_1 = pd.read_csv("datasets/synthetic-1.csv", delimiter = ",", names = ["x", "y", "class"])
    synthetic_data_df_2 = pd.read_csv("datasets/synthetic-2.csv", delimiter = ",", names = ["x", "y", "class"])
    synthetic_data_df_3 = pd.read_csv("datasets/synthetic-3.csv", delimiter = ",", names = ["x", "y", "class"])
    synthetic_data_df_4 = pd.read_csv("datasets/synthetic-4.csv", delimiter = ",", names = ["x", "y", "class"])

    # Storing all synthetic data in a list
    raw_synthetic_dataset_list = [synthetic_data_df_1, synthetic_data_df_2, synthetic_data_df_3, synthetic_data_df_4]

    # Copying data to preserve original for testing
    synthetic_dataset_list = [synthetic_data_df_1.copy(), synthetic_data_df_2.copy(), synthetic_data_df_3.copy(), synthetic_data_df_4.copy()]
    
    # Discretizing the synthetic dataset
    for dataset in synthetic_dataset_list:
        dataset["x"] = pd.qcut(dataset["x"], NUM_BINS)
        dataset["y"] = pd.qcut(dataset["y"], NUM_BINS)

    # Parameters for the synthetic data
    synthetic_dataset_trees = []
    synthetic_attributes = ["x", "y"]

    # Training, printing, and on the synthetic data
    for i in range(len(synthetic_dataset_list)):
        synthetic_dataset_trees.append(Decision_Tree(synthetic_dataset_list[i], SYNTHETIC_CLASS_LABEL, synthetic_attributes))

        print("Printing Synthetic Tree " + str(i+1) + ":")
        print(synthetic_dataset_trees[i])
        
    # Testing on synthetic data and printing accuracies
    for i in range(len(synthetic_dataset_trees)):
        accuracy = synthetic_dataset_trees[i].test_on_tree(synthetic_dataset_list[i], SYNTHETIC_CLASS_LABEL)

        print("Synthetic" + str(i+1) + "Test Accuracy: " + str(accuracy))

    
    # PART 2 of Assignment ---------------------------------------------------------------------------------------------------
    #   Used https://scikit-learn.org/stable/auto_examples/tree/plot_iris_dtc.html as reference
    #   Used https://matplotlib.org/stable/api/pyplot_summary.html for reference

    # Plot parameters
    plot_step = 0.1
    subplot_titles = ["Synthetic-1", "Synthetic-2", "Synthetic-3", "Synthetic-4"]

    # Current dataset index being looked at (index = dataset # - 1)
    index = 0

    # Need four subplots (one per dataset)
    figure, axs = plt.subplots(2, 2, figsize=(30,15))

    for ax in axs.flat:

        # Current dataset being looked at
        dataset = raw_synthetic_dataset_list[index]

        # Getting plot mins and maxes and modifying plot parameters
        x_min, x_max = dataset["x"].min() - 1, dataset["x"].max() + 1
        y_min, y_max = dataset["y"].min() - 1, dataset["y"].max() + 1

        # Creating meshgrid
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), 
                             np.arange(y_min, y_max, plot_step))
        
        # test_on_tree() is expecting a DataFrame object
        data = np.c_[xx.ravel(), yy.ravel()]
        meshgrid_df = pd.DataFrame(data, columns = synthetic_attributes)

        # Obtaining predictions
        z = synthetic_dataset_trees[index].test_raw_data(meshgrid_df, SYNTHETIC_CLASS_LABEL)
        z = np.reshape(z, xx.shape)

        # Plotting the approximation background
        cs = ax.contourf(xx, yy, z, levels = 1, colors = ['salmon', 'cornflowerblue'])
        #ax.axis("tight")

        # Labeling the axis
        plt.xlabel("x", fontsize = 20)
        plt.ylabel("y", fontsize = 20)

        # Plotting data
        ax.scatter(dataset.loc[dataset["class"] == 0]["x"], dataset.loc[dataset["class"] == 0]["y"], c = 'r', label = "0", cmap = plt.cm.Paired, edgecolor = "black", s = 25)
        ax.scatter(dataset.loc[dataset["class"] == 1]["x"], dataset.loc[dataset["class"] == 1]["y"], c = 'b', label = "1", cmap = plt.cm.Paired, edgecolor = "black", s = 25)
        
        # Formatting the subplot
        ax.axis("tight")
        ax.set_title(subplot_titles[index], fontsize = 20)
        ax.legend(fontsize = 15)

        # Need to move on to next dataset next loop iteration
        index += 1
    
    # Formatting the plot
    plt.suptitle("Decision Surface of Decision Trees", fontsize = 35)
    figure.savefig("Decision_Surface.png")
# synthetic_data()


# Description: Train and test the Decision Tree for the pokemon data
# Arguments: None
# Returns: None
def pokemon_data():

    # Reading in pokemon data
    pokemon_stats = pd.read_csv("datasets/pokemonStats.csv", delimiter = ",")
    pokemon_legendary = pd.read_csv("datasets/pokemonLegendary.csv", delimiter = ",")

    # Combining the two DataFrames
    pokemon_dataset = pd.concat([pokemon_stats, pokemon_legendary], axis = 1)

    # Obtaining the attributes
    pokemon_attributes = pokemon_dataset.columns.tolist()
    pokemon_attributes.remove(POKEMON_CLASS_LABEL)

    # Attributes that need to be discretized
    continuous_attributes = ["Total", "HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]

    # Discretizing the pokemon dataset
    for attribute in continuous_attributes:
        pokemon_dataset[attribute] = pd.qcut(pokemon_dataset[attribute], NUM_BINS)

    # Training on the pokemon data
    pokemon_tree = Decision_Tree(pokemon_dataset, POKEMON_CLASS_LABEL, pokemon_attributes)

    # Printing the pokemon tree
    print("Printing Pokemon Tree:")
    print(pokemon_tree)
    
    # Testing on pokemon data and printing accuracies
    accuracy = pokemon_tree.test_on_tree(pokemon_dataset, POKEMON_CLASS_LABEL)
    print("Pokemon Test Accuracy: " + str(accuracy))
# pokemon_data()


# Description: Decision_Trees are objects. Neater code this way
class Decision_Tree:

    # Description: Decision tree initialization. Creates tree and sets root node
    def __init__(self, dataset, class_label, attributes):
        self.root_node = self.ID3(dataset, class_label, attributes, 0)
        

    # Overload printing
    def __str__(self):
        self.print_tree(self.root_node, 0)
        return ""
    # str()


    # Description: Print tree in nice format
    # Arguments: root of current tree being printed, depth currently at
    # Returns: Accuracy of the test
    def print_tree(self, root, depth):
        
        # Formnatting line based on current depth
        for i in range(depth):
            print("   ", end='')

        # Printing the current node
        print(str(root))
        
        # Recursing through each child
        for child in root.child_nodes:
            self.print_tree(child, depth+1)
    # print_tree()


    # Description: Main decision tree creation function.
    # Arguments: dataset (examples), class label for the dataset, array of attribute objects
    # Returns: Root of the current tree (subtree starting at root)
    def ID3(self, dataset, class_label, attributes, depth):

        root = Node()

        # Checking if there is only one type of class label left
        if len(dataset[class_label].unique()) == 1 or len(attributes) == 0 or depth == 3:
            root.attribute = dataset[class_label].value_counts().idxmax()
            return root
              
        # Finding best attribute to split on
        splitting_attribute = self.best_attribute(dataset, class_label, attributes)
        
        # Setting root attribute
        root.attribute = splitting_attribute

        # Getting unique values
        if type(dataset[splitting_attribute].values) == 'pandas.core.arrays.categorical.Categorical':
            unique_values = dataset[splitting_attribute].values.unique()
        else:
            unique_values = np.unique(dataset[splitting_attribute].values)

        # Cycling through attribute values are creating branches
        for attribute_value in unique_values:
            
            # Creating subset of dataset with current attribute_value
            subset = dataset.loc[dataset[splitting_attribute] == attribute_value]

            # If subset is empty, return most common class label
            if len(subset) == 0:
                new_node = Node(attribute_value)
                new_node.attribute = dataset[class_label].value_counts().idxmax()
                root.child_nodes.append(new_node)
            
            # Otherwise begin branch/child creation
            else:
                
                # Creating new attribute list without splitting attribute
                new_attributes = []
                for attribute in attributes:
                    if attribute != root.attribute:
                        new_attributes.append(attribute)
                
                new_node = self.ID3(subset, class_label, new_attributes, depth+1)
                new_node.value = attribute_value
                root.child_nodes.append(new_node)
    
        return root
    # ID3()


    # Description: Testing a dataset on the tree
    # Arguments: dataset being tested
    # Returns: Accuracy of the test
    def test_on_tree(self, test_data, class_label):
        num_correct_predicts = 0

        for index, data in test_data.iterrows():
            if data[class_label] == self.predict_label(data, self.root_node):
                num_correct_predicts += 1

        return num_correct_predicts/len(test_data)
    # test_on_tree()


    # Description: Predict the class label using the decision tree
    # Arguments: Data being tested on
    # Returns: Predicted label
    def predict_label(self, data, root):
        
        # If this node is a leaf, return the label
        if not root.child_nodes:
            return root.attribute
        
        # Going deeper into the tree
        for child in root.child_nodes:
            if child.value == data[root.attribute]:
                return self.predict_label(data, child)                
    # predict_label()


    # Description: Testing raw, un-binned synthesized data
    # Arguments: Data being tested on and class_label
    # Returns: Array of predictions
    def test_raw_data(self, dataset, class_label):
      
        # Store return value
        predictions = []

        # Predicting each piece of data and storing result
        for index, data in dataset.iterrows():
            prediction = self.predict_label_raw(data, self.root_node)

            if str(prediction) == "None":
                # For better looking plot
                predictions.append(0)
            else:
                predictions.append(self.predict_label_raw(data, self.root_node))

        return predictions
    # test_raw_data()

    # Description: Predict the class label using the decision tree
    # Arguments: Data being tested on
    # Returns: Predicted label
    def predict_label_raw(self, data, root):
        
        # If this node is a leaf, return the label
        if not root.child_nodes:
            return root.attribute
        
        # Going deeper into the tree
        for child in root.child_nodes:
            if data[root.attribute] in child.value:
                return self.predict_label_raw(data, child)
    # predict_label_raw()


    # Description: Selects the best attribute to split on at this position in the tree
    # Arguments: dataset (examples), class label for the dataset, array of attribute objects
    # Returns: The best attribute to split on
    def best_attribute(self, dataset, class_label, attributes):

        best_attribute = ""
        best_info_gain = 0

        # Cycling through all the attributes and calculating the information gain
        for attribute in attributes:
            curr_info_gain = self.information_gain(dataset, class_label, attribute)

            # If the information gain 
            if curr_info_gain > best_info_gain:
                best_info_gain = curr_info_gain
                best_attribute = attribute

        return best_attribute
    # best_attribute()


    # Description: Calculate information gain for splitting on an attribute
    # Arguments: dataset (examples), class label for the dataset, attribute splitting on
    # Returns: Information gain value for given attribute
    def information_gain(self, dataset, class_label, chosen_attribute):
    
        # Track number of occurrences of each value of the chosen attribute in the dataset
        attribute_value_count = {}

        # Filling in dict
        if type(dataset[chosen_attribute].values) == 'pandas.core.arrays.categorical.Categorical':
            for attribute_value in dataset[chosen_attribute].values.unique():
                attribute_value_count[attribute_value] = 0
        else:
            for attribute_value in np.unique(dataset[chosen_attribute].values):
                attribute_value_count[attribute_value] = 0

        # Tallying occurrences
        for index, data in dataset.iterrows():
            attribute_value_count[data[chosen_attribute]] += 1

        #Calculating Entropy
        average_child_entropy = 0

        for value in attribute_value_count:
            # Obtaining subset of dataset split on chosen attribute
            new_dataset = dataset.loc[dataset[chosen_attribute] == value]

            average_child_entropy += (attribute_value_count[value]/len(dataset)) * self.entropy(new_dataset, class_label)

        return self.entropy(dataset, class_label) - average_child_entropy
    # information_gain()


    # Description: Splits a dataset based on the value of a given attribute
    # Arguments: dataset (examples), attribute splitting on, attribute value wanted
    # Returns: New dataset split on given attribute
    def split_dataset(self, dataset, chosen_attribute, chosen_attribute_value):
        
        new_dataset = []
        
        #Spliting dataset based on given attribute and storing data with chosen_attribute_value
        for data in dataset:
            if data.attribute_values[chosen_attribute.attribute_name] == chosen_attribute_value:
                new_dataset.append(data)

        return new_dataset
    # split_dataset()


    # Description: Calculates entropy for a dataset
    # Arguments: dataset (examples), class label for the dataset
    # Returns: Entropy of dataset
    def entropy(self, dataset, class_label):   
        
        # Stores numbers of positive/negative class label occurrences
        num_positive = 0
        num_negative = 0
        
       # Tally up occurrences for each class_label value
        for index, data in dataset.iterrows():
            # Adding an occurrence to the current data's class_label_value
            if data[class_label] == 1:
                num_positive += 1
            else:
                num_negative += 1 

        # Case when all labels are the same
        if num_positive == 0 or num_negative == 0:
            return 0
        
        # Calculating positive and negative class_label parts of entropy calculation
        positive = (-num_positive/len(dataset)) * (math.log(num_positive/len(dataset) , 2))
        negative = (-num_negative/len(dataset)) * (math.log(num_negative/len(dataset) , 2))

        # Returning positive part - negative part
        return positive + negative
    # entropy()


main()