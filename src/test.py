from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.metrics.pairwise import cosine_similarity
import operator
from nltk.stem.porter import PorterStemmer
#from nltk.stem import WordNetLemmatizer
import random
from sklearn.metrics.pairwise import euclidean_distances

stemmer = PorterStemmer()												#Porter Stemmer method for stemming the words
#wordnet_lemma = WordNetLemmatizer()
vectorizer = TfidfVectorizer(stop_words='english')						#TF-IDF Vectorizer to create list of feature vectors
tokenizer = RegexpTokenizer(r'\w+')										#tokenizer to create tokens

test_len = 1200															#size of test set
total_len = 4000														#size of training set

print("Reading data from trainhw1.txt and testdatahw1.txt")

ftrain = open("trainhw1.txt")											#read training data
train_data = ftrain.readlines()
ftrain.close()

fbag = open("bag_of_words.txt")											#read bag of words
bag_of_words = fbag.read().split(" ")
fbag.close()

Y = [""]*(total_len)

print("Now preprocessing the data")

corpus = []
random_data = [""]*total_len
index = 0
for i in random.sample(range(0, len(train_data)), total_len):			#select a sample from training data
    random_data[index] = (train_data[i])
    index += 1

for i in range(total_len):												#traverse data and append stemmed words to corpus
    new_list = []
    if random_data[i][0] == "-":
        Y[i] = "-1"
    else:
        Y[i] = "+1"
    list = tokenizer.tokenize(random_data[i][3:])
    for word in list:
        if word.lower() in bag_of_words:
            new_list.append(stemmer.stem(word.lower()))
            #new_list.append(wordnet_lemma.lemmatize(word.lower()))
    corpus.append(" ".join(new_list) + "\n")

print("Creating feature vectors")

X = vectorizer.fit_transform(corpus)									#creates list of feature vectors

Z = X.toarray()

correct = 0
incorrect = 0

print("Applying K Nearest Neighbour Algorithm")

for k in range(300):
    for i in range(test_len):											#traverse feature vectors of test data
        distance = []
        for j in range(test_len, total_len):							#traverse feature vectors of training data
            if i == j:
                continue
            dist = euclidean_distances(Z[j].reshape(1, -1), Z[i].reshape(1, -1))
            #dist = cosine_similarity(Z[j].reshape(1, -1), Z[i].reshape(1, -1))
            distance.append((dist, Y[j]))

        distance.sort(key=operator.itemgetter(0))						#sort the feature vectors in increasing order of Euclidean Distance

        countpos = 0
        countneg = 0
        for j in range(2*k + 1):										#to identify the majority sentiment + or -
            if distance[j][1] == "+1":
                countpos += 1
            else:
                countneg += 1

        if countpos > countneg:
            type = "+1"
        else:
            type = "-1"

        if type == Y[i]:												#to check if predicted sentiment matches the actual sentiment
            correct += 1
        else:
            incorrect += 1

    print("Accuracy for k = " + str(2*k + 1))
    print((correct * 100)/(correct + incorrect))						#calculates and prints the accuracy for given K
