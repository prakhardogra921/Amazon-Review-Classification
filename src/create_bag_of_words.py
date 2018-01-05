from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

tokenizer = RegexpTokenizer(r'\w+')											#tokenizer to create a list of tokens
stop_words = list(stopwords.words("english"))								#list of stopwords

def check(w):																#function to neglect stop words
    if w in stop_words:
        return False
    return True

f = open("trainhw1.txt")
data = f.readlines()
d = {}
for line in data:															#read training data line by line and create a dictionary
    for word in tokenizer.tokenize(line[3:]):
        if(check(word)):
            d[word.lower()] = d.get(word.lower(), 0) + 1
f.close()

f = open("testdatahw1.txt")
test_data = f.readlines()
for line in test_data:														#read test data line by line and add words to the dictionary
    for word in tokenizer.tokenize(line):
        if(check(word)):
            d[word.lower()] = d.get(word.lower(), 0) + 1
f.close()

f = open("bag_of_words_smaller.txt", "a+")
for key, value in d.items():												#write dictionary words to a txt file
    if len(key) > 2:
        if value > 60:
            f.write(key + " ")
f.close()





