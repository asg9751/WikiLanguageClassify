"""
Author: Amol Gaikwad

This program perform language classification between English and Dutch words.

"""
import sys
import pickle
import math


def get_count_classes(classes, range_classes):
    """
    Get count of each class
    :param classes: class values
           range_classes: range of class values

    :return: entropy value
    """
    count_en = len([idx for idx in range_classes if classes[idx] == 'en'])
    count_nl = len([idx for idx in range_classes if classes[idx] == 'nl'])

    return count_en, count_nl

def entropy(val):
    """
        Entropy function
        :param value: input val
        :return: entropy value
    """
    if val == 1:
        return 0
    return (-1) * (val * math.log(val, 2.0) + (1 - val) * math.log((1 - val), 2.0))

def train_dtree(root, attributes, classes, range_classes, visited, depth, prev):
    """
    Build tree by appending right and left child for a node for a given depth

    :param root: Current node
    :param attributes: Set of attribute values
    :param classes: class values
    :param range_classes: range of class values
    :param visited: Visited nodes
    :param depth: Level to consider
    :param prev: Earlier prediction

    :return:None
    """


    count_en, count_nl = get_count_classes(classes, range_classes)
    # If depth is reached return the max
    if depth == len(attributes) - 1 or len(attributes) == len(visited):
        if count_en > count_nl:
            root.value = 'en'
        else:
            root.value = 'nl'

    # If examples over return prev prediction
    elif len(range_classes) == 0:
        root.value = prev

    # If only samples of one class remain
    elif [classes[index] for index in range_classes].count(classes[range_classes[0]]) == len(range_classes):
        root.value = classes[range_classes[0]]


    # Find best attr to split
    else:
        gain = []


        # For all attributes
        for index_attribute in range(len(attributes)):

            # Check if already splitted
            if index_attribute in visited:
                gain.append(0)
                continue

            # Find best splitting attribute
            else:

                count_true_en = len([idx for idx in range_classes if classes[idx] == 'en' and attributes[index_attribute][idx]])
                count_true_nl = len([idx for idx in range_classes if classes[idx] == 'nl' and attributes[index_attribute][idx]])
                count_false_en = len([idx for idx in range_classes if classes[idx] == 'en' and not attributes[index_attribute][idx]])
                count_false_nl = len([idx for idx in range_classes if classes[idx] == 'nl' and not attributes[index_attribute][idx]])

                # Dont split if samples are from one class
                if (count_true_nl + count_true_en == 0) or (count_false_en + count_false_nl == 0):
                    gain.append(0)
                    continue

                if count_true_en == 0 and count_false_en == 0:
                    entropy_child_true = 0
                    entropy_child_false = 0
                elif count_true_en == 0:
                    entropy_child_true = 0
                    entropy_child_false = ((count_false_en + count_false_nl) / (count_nl + count_en)) \
                                      * entropy(count_false_en / (count_false_nl + count_false_en))
                elif count_false_en == 0:
                    entropy_child_false = 0
                    entropy_child_true = ((count_true_en + count_true_nl) / (count_nl + count_en)) \
                                     * entropy(count_true_en / (count_true_nl + count_true_en))
                else:
                    entropy_child_true = ((count_true_en + count_true_nl) / (count_nl + count_en)) \
                                     * entropy(count_true_en / (count_true_nl + count_true_en))

                    entropy_child_false = ((count_false_en + count_false_nl) / (count_nl + count_en)) \
                                      * entropy(count_false_en / (count_false_nl + count_false_en))

                # Find the gain for each attribute
                entropy_parent = entropy(count_en / (count_en + count_nl))
                entropy_child = entropy_child_true + entropy_child_false
                attr_gain = entropy_parent - entropy_child
                gain.append(attr_gain)

        # If max gain is 0 then return
        if max(gain) == 0:
            root.value = prev
            return

        # Select the max gain attribute
        max_gain_attr = gain.index(max(gain))

        visited.append(max_gain_attr)

        # Split for max gain attribute
        max_gain_true = [idx for idx in range_classes if attributes[max_gain_attr][idx]]
        max_gain_false = [idx for idx in range_classes if not attributes[max_gain_attr][idx]]


        # Current prediction
        if count_en > count_nl:
            curr_prediction = 'en'
        else:
            curr_prediction = 'nl'

        root.value = max_gain_attr

        # Make left node for the max gain attribute

        left_node = Node()
        # Make right node for the max gain attribute

        right_node = Node()

        root.left = left_node
        root.right = right_node
        # Recurse left and right half
        train_dtree(left_node, attributes, classes, max_gain_true, visited, depth + 1, curr_prediction)
        train_dtree(right_node, attributes, classes, max_gain_false, visited, depth + 1, curr_prediction)

        del visited[-1]


def predict_dtree(test_file, model_file):
    """
        Prediction for the decision tree
        :param test_file: input test file
        :param model_file: Model file

        :return:None
    """

    # Load model
    root = pickle.load(open(model_file, 'rb'))
    file = open(test_file)
    sentence_list = []

    for line in file:
        words = line.split()
        sentence_list.append(" ".join(words))

    attributes = []
    num_features = 11

    for count in range(num_features):
        attributes.append([])

    # Fill attribute values
    for line in sentence_list:
        attributes[0].append(contains_q(line))
        attributes[1].append(contains_x(line))
        attributes[2].append(check_avg_word_len(line))
        attributes[3].append(contains_van(line))
        attributes[4].append(contains_de_het(line))
        attributes[5].append(contains_een(line))
        attributes[6].append(contains_en(line))
        attributes[7].append(contains_common_dutch_words(line))
        attributes[8].append(contains_common_eng_words(line))
        attributes[9].append(contains_a_an_the(line))
        attributes[10].append(contains_and(line))


    # Find prediction for every line by passing it through a tree
    for line_idx in range(len(sentence_list)):
        node = root
        while type(node.value) != str:
            val = attributes[node.value][line_idx]
            if val is True:
                node = node.left
            else:
                node = node.right
        print(node.value)

def predict_adaboost(test_file, model_file):
    """
    Predict using adaboost
    :param test_file: input test file
    :param model_file: Model file

    :return:None
    """
    # Load model
    node = pickle.load(open(model_file, 'rb'))
    file = open(test_file)
    sentence_list = []

    # Process words
    for line in file:
        words = line.split()
        sentence_list.append(" ".join(words))
    attributes = []
    num_features = 11

    for count in range(num_features):
        attributes.append([])
    # Fill attribute values
    for line in sentence_list:
        attributes[0].append(contains_q(line))
        attributes[1].append(contains_x(line))
        attributes[2].append(check_avg_word_len(line))
        attributes[3].append(contains_van(line))
        attributes[4].append(contains_de_het(line))
        attributes[5].append(contains_een(line))
        attributes[6].append(contains_en(line))
        attributes[7].append(contains_common_dutch_words(line))
        attributes[8].append(contains_common_eng_words(line))
        attributes[9].append(contains_a_an_the(line))
        attributes[10].append(contains_and(line))

    # Predict using adaboost
    for line_idx in range(len(sentence_list)):
        sum = 0
        for index in range(len(node[0])):
            sum += predict_final(node[0][index], attributes, line_idx) * node[1][index]

        if sum > 0:
            print('en')
        else:
            print('nl')

def predict_final(stump, attributes, index):
    """
    Classifier prediction
    :param stump: input stump
    :param attributes: attribute values
    :param index: index of test sentence

    :return:
    """
    if attributes[stump.value][index]:
        if stump.left.value == 'en':
            return 1
        else:
            return -1
    else:
        if stump.right.value == 'en':
            return 1
        else:
            return -1

def get_attributes_tree(training_file, model_file):
    """
    Extract training samples and build model using decison tree

    :param training_file:Training file
    :param model_file: File to which model is to be written
    :return:None
    """

    classes, sentences = extract_data(training_file)
    print("Number of training samples "+str(len(classes)))
    attributes = []
    num_features = 11

    for count in range(num_features):
        attributes.append([])

    # For each line set the values for features for that line
    for line in sentences:
        attributes[0].append(contains_q(line))
        attributes[1].append(contains_x(line))
        attributes[2].append(check_avg_word_len(line))
        attributes[3].append(contains_van(line))
        attributes[4].append(contains_de_het(line))
        attributes[5].append(contains_een(line))
        attributes[6].append(contains_en(line))
        attributes[7].append(contains_common_dutch_words(line))
        attributes[8].append(contains_common_eng_words(line))
        attributes[9].append(contains_a_an_the(line))
        attributes[10].append(contains_and(line))


    range_classes = range(len(classes))
    # Track attributes splitted
    visited = []
    root = Node()

    # Build decision tree
    train_dtree(root, attributes, classes, range_classes, visited, 0, None)

    # Dump hypothesis to a file using pickle
    fp = open(model_file, 'wb')
    pickle.dump(root, fp)

def get_attributes_adaboost(training_file, model_file, number_of_decision_stumps):
    """
    Extract training samples and build model using adaboost

    :param training_file:Training file
    :param model_file: File to which model is to be written
    :param number_of_decision_stumps: number of stumps in consideration

    :return:None
    """
    classes, sentences = extract_data(training_file)
    weights = [1 / len(sentences)] * len(sentences)

    attributes = []
    num_features = 11

    for count in range(num_features):
        attributes.append([])

    # For each line set the values for features for that line
    for line in sentences:
        attributes[0].append(contains_q(line))
        attributes[1].append(contains_x(line))
        attributes[2].append(check_avg_word_len(line))
        attributes[3].append(contains_van(line))
        attributes[4].append(contains_de_het(line))
        attributes[5].append(contains_een(line))
        attributes[6].append(contains_en(line))
        attributes[7].append(contains_common_dutch_words(line))
        attributes[8].append(contains_common_eng_words(line))
        attributes[9].append(contains_a_an_the(line))
        attributes[10].append(contains_and(line))

    stumps = []

    classifier_weights = [1] * number_of_decision_stumps

    range_classes = range(len(classes))


    # Build usinf adaboost algorithm
    for count in range(number_of_decision_stumps):

        root = Node()
        # Get stump
        stump = return_stump(root, attributes, classes, range_classes, weights)
        error = 0

        for index in range(len(sentences)):

            # Calculate error for incorrect values
            if stump_prediction(stump, attributes, index) != classes[index]:
                error = error + weights[index]

        for index in range(len(sentences)):

            # Update weights
            if stump_prediction(stump, attributes, index) == classes[index]:
                weights[index] = weights[index] * error / (1 - error)

        sum_weights = sum(weights)
        # Normalize weights
        for idx in range(len(weights)):
            weights[idx] = weights[idx] / sum_weights

        # Updated hypothseis weight
        classifier_weights[count] = math.log(((1 - error)/(error)), 2)
        stumps.append(stump)

    # Dump hypothesis to a file using pickle
    fp = open(model_file, 'wb')
    pickle.dump((stumps, classifier_weights), fp)

def return_stump(root, attributes, classes, range_classes, weights):
    """
    Function returns a decision stump
    :param depth:Depth of the tree we are at
    :param root:
    :param attributes:
    :param results:
    :param total_results:
    :param weights:
    :return:
    """
    gain = []
    count_en = 0
    count_nl = 0
    for index in range_classes:
        if classes[index] == 'en':
            count_en = count_en + weights[index]
        else:
            count_nl = count_nl + weights[index]

    for index_attribute in range(len(attributes)):

        count_true_en = sum([weights[idx] for idx in range_classes if classes[idx] == 'en' and attributes[index_attribute][idx]])
        count_true_nl = sum([weights[idx] for idx in range_classes if classes[idx] == 'nl' and attributes[index_attribute][idx]])
        count_false_en = sum([weights[idx] for idx in range_classes if classes[idx] == 'en' and not attributes[index_attribute][idx]])
        count_false_nl = sum([weights[idx] for idx in range_classes if classes[idx] == 'nl' and not attributes[index_attribute][idx]])

        if (count_true_nl + count_true_en == 0) or (count_false_en + count_false_nl == 0):
            gain.append(0)
            continue

        if count_true_en == 0 and count_false_en == 0:
            entropy_child_true = 0
            entropy_child_false = 0
        elif count_true_en == 0:
            entropy_child_true = 0
            entropy_child_false = ((count_false_en + count_false_nl) / (count_nl + count_en)) \
                                  * entropy(count_false_en / (count_false_nl + count_false_en))
        elif count_false_en == 0:
            entropy_child_false = 0
            entropy_child_true = ((count_true_en + count_true_nl) / (count_nl + count_en)) \
                                 * entropy(count_true_en / (count_true_nl + count_true_en))
        else:
            entropy_child_true = ((count_true_en + count_true_nl) / (count_nl + count_en)) \
                                 * entropy(count_true_en / (count_true_nl + count_true_en))

            entropy_child_false = ((count_false_en + count_false_nl) / (count_nl + count_en)) \
                                  * entropy(count_false_en / (count_false_nl + count_false_en))

        # Find the gain for each attribute
        entropy_parent = entropy(count_en / (count_en + count_nl))
        entropy_child = entropy_child_true + entropy_child_false
        attr_gain = entropy_parent - entropy_child

        gain.append(attr_gain)

    max_gain_attr = gain.index(max(gain))
    root.value = max_gain_attr

    count_max_true_en = sum([weights[idx] for idx in range(len(attributes[max_gain_attr])) if classes[idx] == 'en' and attributes[max_gain_attr][idx]])
    count_max_true_nl = sum([weights[idx] for idx in range(len(attributes[max_gain_attr])) if classes[idx] == 'nl' and attributes[max_gain_attr][idx]])
    count_max_false_en = sum([weights[idx] for idx in range(len(attributes[max_gain_attr])) if classes[idx] == 'en' and not attributes[max_gain_attr][idx]])
    count_max_false_nl = sum([weights[idx] for idx in range(len(attributes[max_gain_attr])) if classes[idx] == 'nl' and not attributes[max_gain_attr][idx]])

    left_node = Node()
    right_node = Node()

    if count_max_true_en > count_max_true_nl:
        left_node.value = 'en'
    else:
        left_node.value = 'nl'

    if count_max_false_en > count_max_false_nl:
        right_node.value = 'en'
    else:
        right_node.value = 'nl'

    root.left = left_node
    root.right = right_node

    return root

def stump_prediction(stump, attributes, index):
    """
    Stump prediction
    :param stump: input decision stump
    :param attributes: set of attributes
    :param index: index of the statement

    :return: Stump prediction
    """
    if attributes[stump.value][index]:
        return stump.left.value
    else:
        return stump.right.value

def extract_data(file):
    """
    Separate class and sentences for training
    :param file:input training file
    :return: classes: final classes
             sentences: training data
    """

    classes = []
    sentences = []
    with open(file, encoding="utf-8") as f:
        for line in f:
            text = line.split('|')
            classes.append(text[0])
            sentences.append(text[1].strip())

    print(classes)
    print(sentences)

    return classes, sentences

def check_avg_word_len(line):
    """
    Check the average word length of the statement
    :param line: input statement
    :return: True if average word size is greater than 5, False otherwise
    """
    words = line.split()
    sum_word_size = 0
    count_words = len(words)
    for word in words:
        sum_word_size = sum_word_size + len(word)
    if sum_word_size / count_words > 5:
        return True
    else:
        return False

def contains_common_dutch_words(line):
    """
    Check if sentence contains common dutch words
    :param line: input statement
    :return: True if common dutch words present, False otherwise
    """
    list = ['naar','deze','ons','meest','voor','niet','met','hij','zijn','be','ik','het','ze','wij','hem','weten'
            'jouw','dan','ook','onze','ze','er','hun','zo','over']
    words = line.split()
    for word in words:
        word = word.lower().replace(',', '')
        if word in list:
            return True
    return False

def contains_common_eng_words(line):
    """
    Check if sentence contains common english words
    :param line: input statement
    :return: True if common english words present, False otherwise
    """
    list = ['I', 'it','our','these','us''me','for','not','with','most','they','we','she','there', 'their','so', 'about','he','his','to','be',
            'him','know','your','than','then','also']
    words = line.split()
    for word in words:
        word = word.lower().replace(',', '')
        if word in list:
            return True
    return False

def contains_en(line):
    """
        Check if sentence contains 'en'
        :param line: input statement
        :return: True if 'en' present, False otherwise
    """
    words = line.split()
    for word in words:
        word = word.lower().replace(',', '')
        if word == 'en':
            return True
    return False



def contains_van(line):
    """
        Check if sentence contains 'van'
        :param line: input statement
        :return: True if 'van' present, False otherwise
    """
    words = line.split()
    for word in words:
        word = word.lower().replace(',', '')
        if word == 'van':
            return True
    return False


def contains_een(line):
    """
    Check if sentence contains 'een'
    :param line: input statement
    :return: True if 'een' present, False otherwise
    """
    words = line.split()
    for word in words:
        word = word.lower().replace(',','')
        if word == 'een':
            return True
    return False

def contains_de_het(line):
    """
    Check if sentence contains 'de' or 'het'
    :param line: input statement
    :return: True if 'de' or 'het' present, False otherwise
    """
    words = line.split()
    for word in words:
        word = word.lower().replace(',', '')
        if word == 'de' or word == 'het':
            return True
    return False

def contains_a_an_the(line):
    """
    Check if sentence contains a, an or the
    :param line: input statement
    :return: True if 'a', 'an' or 'the' present, False otherwise
    """
    words = line.split()
    for word in words:
        word = word.lower().replace(',', '')
        if word == 'a' or word == 'an' or word == 'the':
            return True
    return False


def contains_and(line):
    """
    Check if sentence contains 'and'
    :param line: input statement
    :return: True if 'and' present, False otherwise
    """
    words = line.split()
    for word in words:
        word = word.lower().replace(',', '')
        if word == 'and':
            return True
    return False


def contains_q(line):
    """
    Check if sentence contains q
    :param line: input statement
    :return: True if q present, False otherwise
    """
    if line.find('Q') < 0 or line.find('q') < 0:
        return False
    else:
        return True

def contains_x(line):
    """
    Check if sentence contains x
    :param line: input statement
    :return: True if x present, False otherwise
    """
    if line.find('x') < 0 or line.find('X') < 0:
        return False
    else:
        return True



def main():
    """
    Main Function
    :return: None
    """
    # Read number of arguments
    noofargs = len(sys.argv)

    # Check for invalid number of arguments
    if (noofargs != 5):
        print("Invalid number of arguments")
        print('Syntax :train <trainingfile> <outputmodelfile> <learningtype(dtree or ada)>')
        print('or')
        print('Syntax :predict <inputfile> <model> <testingtype(dtree or ada)>')

    else:
        num_stumps = 50
        if sys.argv[1] == 'train':
            if sys.argv[4] == 'dtree':
                get_attributes_tree(sys.argv[2], sys.argv[3])
            else:
                get_attributes_adaboost(sys.argv[2], sys.argv[3], num_stumps)
        elif sys.argv[1] == 'predict':
            if sys.argv[4] =='dtree':
                predict_dtree(sys.argv[2], sys.argv[3])
            else:
                predict_adaboost(sys.argv[2], sys.argv[3])


class Node:
    """
        Represents a node in a tree or a stump
    """
    __slots__ = 'value','left','right'

    def __init__(self):
        self.value = None
        self.left = None
        self.right = None

if __name__ == "__main__":
    main()