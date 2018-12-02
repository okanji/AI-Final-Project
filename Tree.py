import math
from Node import Node


def data_proportions(data):
    # Find the proportions
    all_labels = [label for (features, label) in data]
    num_entries = len(all_labels)
    possible_labels = set(all_labels)

    proportions = []
    for label in possible_labels:
        proportions.append(float(all_labels.count(label)) / num_entries)

    return proportions


def entropy(proportions):
    return -sum([p * math.log(p, 2) for p in proportions])


def split_data(data, feature_index):
    # splits the data based on a single feature of index feature_index

    # get possible values of the given feature
    feature_values = [features[feature_index] for (features, label) in data]

    subsets = []

    for feature in set(feature_values):
        # compute the piece of the split corresponding to the chosen value
        data_subset = [(features, label) for (features, label) in data if features[feature_index] == feature]

        subsets.append(data_subset)
        #yield data_subset

    return subsets


def gain(data, feature_index):

    gain = entropy(data_proportions(data))

    for data_subset in split_data(data, feature_index):
        gain -= entropy(data_proportions(data_subset))

    return gain


def labels_all_same(data):
    # return true if all the labels are the same
    return len(set([label for (features, label) in data])) <= 1


def majority(data, node):
    labels = [label for (_, label) in data]
    choice = max(set(labels), key=labels.count)
    node.label = choice

    return node


def make_decision_tree(data, root, remaining_features):
    # Make the decision tree recursively by appending children to a root node

    # First base case, if all the labels are the same and so gain will not be a useful measure to split the data,
    # simply classify the root with that label
    if labels_all_same(data):
        root.label = data[0][1]
        return root

    # If there are no remaining features to split the node on, classify the node with the simple majority
    # of the labels as using simple probability, the node is likely belongs class that is the majority
    if len(remaining_features) == 0:
        return majority(data, root)

    # Find the index of the best feature to split on, by taking the remaining feature with the largest gain, the lambda
    # function computes the gain for each feature in remaining features. The optional key parameter in max allows
    # you so specify the metric for computing max, in this case since we are finding the max of feature objects it is
    # required as max does not know how to find the max of our object.
    best_feature = max(remaining_features, key=lambda index: gain(data, index))

    # If there is no information gain, classify the node with the simple majority
    # of the labels as using simple probability, the node is likely belongs class that is the majority
    if gain(data, best_feature) == 0:
        return majority(data, root)

    root.split_feature = best_feature

    # add child nodes and do this recursively in order to make a tree
    for split in split_data(data, best_feature):
        child_node = Node(parent=root)
        child_node.split_feature_value = split[0][0][best_feature]
        root.children.append(child_node)

        # note, subtracting set a from set b in python removes from set a any
        #  common elements between set a and b and returns this as a new set
        make_decision_tree(split, child_node, remaining_features - set([best_feature]))

    return root


# For the first time we run the decision tree we only pass the data as a parameter as we have not split the data yet
def decision_tree(data):
    return make_decision_tree(data, Node(), set(range(len(data[0][0]))))


def classify(node, features):
    # Classify the decision tree recursively

    if node.children == []:
        return node.label
    else:
        matching_children = [child for child in node.children if child.split_feature_value == features[node.split_feature]]

        return classify(matching_children[0], features)


if __name__ == '__main__':
    with open('house-votes-1984.txt', 'r') as input_file:
        lines = input_file.readlines()

    data = [line.strip().split(',') for line in lines]
    data = [(x[1:], x[0]) for x in data]

    clean_data = [x for x in data if '?' not in x[0]]

    clean_data_len = len(clean_data)
    training_data_len = math.floor(0.7 * clean_data_len)

    training_data = clean_data[:training_data_len]
    testing_data = clean_data[training_data_len:]

    testing_data_len = len(testing_data)

    # Train the decision tree with training data
    tree = decision_tree(training_data)

    predicted_classifications = [classify(tree, elem[0]) for elem in testing_data]
    actual_classifications = [elem[1] for elem in testing_data]

    simmilar_count = 0

    for testing_label, actual_label in zip(predicted_classifications, actual_classifications):
        if testing_label == actual_label:
            simmilar_count += 1

    # print("Proportion of correct classifications: {}".format(simmilar_count/testing_data_len))

    print(training_data[0]›)

    print("")
    print("This is a decision tree trained using votes from the USA house of congress in 1984")
    print("\nA 70%, 30% split was used between training and testing data.")
    print("\nThe decision tree predicted the labels of the testing set with {} percent accuracy".format((simmilar_count/testing_data_len) * 100))

    print("\nThe decision tree will now classify your input as either D (Democrat) or R (Republican)")
    print("\nPlease vote for the following bills with either 'y' (Yes) or 'n' (No).")

    votes = []
    questions = ['handicapped-infants: ', 'water-project-cost-sharing: ', ' adoption-of-the-budget-resolution: ', 'physician-fee-freeze: ',
                 'el-salvador-aid: ', 'religious-groups-in-schools: ', 'anti-satellite-test-ban: ', 'aid-to-nicaraguan-contras: ', 'mx-missile: ',
                 'immigration: ', 'synfuels-corporation-cutback: ', 'education-spending: ', 'superfund-right-to-sue: ', 'crime: ', 'duty-free-exports: ',
                 'export-administration-act-south-africa: ']

    for question in questions:
        vote = input(question)
        votes.append(vote)

    result = classify(tree, votes)

    print("According to the decision tree, you are a: {}".format(result))

