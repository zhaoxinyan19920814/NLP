import math
import nltk
import time

# Constants to be used by you when you fill the functions
START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
MINUS_INFINITY_SENTENCE_LOG_PROB = -1000

# TODO: IMPLEMENT THIS FUNCTION
# Calculates unigram, bigram, and trigram probabilities given a training corpus
# training_corpus: is a list of the sentences. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function outputs three python dictionaries, where the keys are tuples expressing the ngram and the value is the log probability of that ngram
def calc_probabilities(training_corpus):
    unigram_p = {}
    bigram_p = {}
    trigram_p = {}

    word_count = {}
    bigram_count = {}
    trigram_count = {}

    unigrams = []
    bigram_tuples = []
    trigram_tuples = []
     
    for data in training_corpus:
	data_unigrams = data + STOP_SYMBOL
	token_unigrams = data_unigrams.split(' ')
	for i in token_unigrams:
	    if word_count.has_key(i):
		word_count[i] += 1
	    else:
		word_count[i] = 1	
        unigrams += token_unigrams


	data_bigrams = START_SYMBOL +' ' + data + STOP_SYMBOL
	token_bigrams = data_bigrams.split(' ')
	
	for i in list(nltk.bigrams(token_bigrams)):
	    if bigram_count.has_key(i):
		bigram_count[i] += 1
	    else:
		bigram_count[i] = 1
	bigram_tuples += list(nltk.bigrams(token_bigrams))


	data_trigrams = START_SYMBOL+' '+ START_SYMBOL+' ' + data + STOP_SYMBOL
	token_trigrams = data_trigrams.split(' ')

	for i in list(nltk.trigrams(token_trigrams)):
	    if trigram_count.has_key(i):
		trigram_count[i] += 1
	    else:
		trigram_count[i] = 1
	trigram_tuples += list(nltk.trigrams(token_trigrams))

    print len(training_corpus)

    L_unigrams = len(unigrams)
    unigram_p = { (word,) : math.log((float((word_count[word]))/float(L_unigrams)),2) for word in set(unigrams)}
    print len(unigram_p.keys()) 
    print unigram_p[('captain',)]
    print unigram_p[('captain\'s',)]
    print unigram_p[('captaincy',)]

    L_bigrams = len(bigram_tuples)
    for bigram in set(bigram_tuples):
	if bigram[0] == START_SYMBOL:
	    bigram_p[bigram] = math.log((float(bigram_count[bigram])/float(len(training_corpus))),2)
	else:
	    bigram_p[bigram] = math.log((float(bigram_count[bigram])/float(word_count[bigram[0]])), 2)
	   
    print bigram_p[('and','religion')]
    print bigram_p[('and','religious')]
    print bigram_p[('and','religiously')]


    L_trigrams = len(trigram_tuples)
    for trigram in set(trigram_tuples):
	if trigram[0] == START_SYMBOL and trigram[1] == START_SYMBOL:
	    trigram_p[trigram] = math.log((float(trigram_count[trigram])/float(len(training_corpus))),2)
	else:
	    trigram_p[trigram] = math.log((float(trigram_count[trigram])/float(bigram_count[(trigram[0],trigram[1])])),2)

    print trigram_p[('and','not','a')]
    print trigram_p[('and','not','by')]
    print trigram_p[('and','not','come')]
    print unigram_p[('natural',)]
    print bigram_p[('natural','that')]
    print trigram_p[('natural','that','he')]	

    return unigram_p, bigram_p, trigram_p

# Prints the output for q1
# Each input is a python dictionary where keys are a tuple expressing the ngram, and the value is the log probability of that ngram
def q1_output(unigrams, bigrams, trigrams, filename):
    # output probabilities
    outfile = open(filename, 'w')

    unigrams_keys = unigrams.keys()
    unigrams_keys.sort()
    for unigram in unigrams_keys:
        outfile.write('UNIGRAM ' + unigram[0] + ' ' + str(unigrams[unigram]) + '\n')

    bigrams_keys = bigrams.keys()
    bigrams_keys.sort()
    for bigram in bigrams_keys:
        outfile.write('BIGRAM ' + bigram[0] + ' ' + bigram[1]  + ' ' + str(bigrams[bigram]) + '\n')

    trigrams_keys = trigrams.keys()
    trigrams_keys.sort()    
    for trigram in trigrams_keys:
        outfile.write('TRIGRAM ' + trigram[0] + ' ' + trigram[1] + ' ' + trigram[2] + ' ' + str(trigrams[trigram]) + '\n')

    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence
# ngram_p: python dictionary of probabilities of uni-, bi- and trigrams.
# n: size of the ngram you want to use to compute probabilities
# corpus: list of sentences to score. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function must return a python list of scores, where the first element is the score of the first sentence, etc. 
def score(ngram_p, n, corpus):
    scores = []
    for sentence in corpus:
	score = 0.0
	if n == 1:
	    sentence = sentence + STOP_SYMBOL
	    tokens = sentence.split(' ')
	    for token in tokens:
		if ngram_p.has_key((token,)):
	            score += ngram_p[(token,)]
		else:
		    score = MINUS_INFINITY_SENTENCE_LOG_PROB
		    break
    	    scores.append(score)
	if n == 2:
	    sentence = START_SYMBOL + ' ' + sentence + STOP_SYMBOL
	    tokens = sentence.split(' ')
	    for token in nltk.bigrams(tokens):
		if ngram_p.has_key(token):
		    score += ngram_p[token]
		else:
		    score = MINUS_INFINITY_SENTENCE_LOG_PROB
		    break
	    scores.append(score)
	if n == 3:
	    sentence = START_SYMBOL + ' ' + START_SYMBOL + ' ' + sentence + STOP_SYMBOL
	    tokens = sentence.split(' ')
	    for token in nltk.trigrams(tokens):
		if ngram_p.has_key(token):
		    score += ngram_p[token]
		else:
		    score = MINUS_INFINITY_SENTENCE_LOG_PROB
		    break
	    scores.append(score)
	   
    return scores

# Outputs a score to a file
# scores: list of scores
# filename: is the output file name
def score_output(scores, filename):
    outfile = open(filename, 'w')
    for score in scores:
        outfile.write(str(score) + '\n')
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence with a linearly interpolated model
# Each ngram argument is a python dictionary where the keys are tuples that express an ngram and the value is the log probability of that ngram
# Like score(), this function returns a python list of scores
def linearscore(unigrams, bigrams, trigrams, corpus):
    scores = []

    for sentence in corpus:
	sentence = START_SYMBOL + ' ' + START_SYMBOL + ' ' + sentence + STOP_SYMBOL
	tokens = sentence.split(' ')
	score_sentence = 0.0
	
	for token in nltk.trigrams(tokens):
	    if trigrams.has_key(token):
		t = trigrams[token]
		b = bigrams[(token[1],token[2])]
		u = unigrams[(token[2],)]
		temp = 2**t + 2**b + 2**u
		temp = math.log(((float(1)/float(3))*temp),2)
		score_sentence += temp
	    else:
		score_sentence = MINUS_INFINITY_SENTENCE_LOG_PROB
		break
	
	scores.append(score_sentence)
    return scores

DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'

# DO NOT MODIFY THE MAIN FUNCTION
def main():
    # start timer
    time.clock()

    # get data
    infile = open(DATA_PATH + 'Brown_train.txt', 'r')
    corpus = infile.readlines()
    infile.close()

    # calculate ngram probabilities (question 1)
    unigrams, bigrams, trigrams = calc_probabilities(corpus)

    # question 1 output
    q1_output(unigrams, bigrams, trigrams, OUTPUT_PATH + 'A1.txt')

    # score sentences (question 2)
    uniscores = score(unigrams, 1, corpus)
    biscores = score(bigrams, 2, corpus)
    triscores = score(trigrams, 3, corpus)

    # question 2 output
    score_output(uniscores, OUTPUT_PATH + 'A2.uni.txt')
    score_output(biscores, OUTPUT_PATH + 'A2.bi.txt')
    score_output(triscores, OUTPUT_PATH + 'A2.tri.txt')

    # linear interpolation (question 3)
    linearscores = linearscore(unigrams, bigrams, trigrams, corpus)

    # question 3 output
    score_output(linearscores, OUTPUT_PATH + 'A3.txt')

    # open Sample1 and Sample2 (question 5)
    infile = open(DATA_PATH + 'Sample1.txt', 'r')
    sample1 = infile.readlines()
    infile.close()
    infile = open(DATA_PATH + 'Sample2.txt', 'r')
    sample2 = infile.readlines()
    infile.close() 

    # score the samples
    sample1scores = linearscore(unigrams, bigrams, trigrams, sample1)
    sample2scores = linearscore(unigrams, bigrams, trigrams, sample2)

    # question 5 output
    score_output(sample1scores, OUTPUT_PATH + 'Sample1_scored.txt')
    score_output(sample2scores, OUTPUT_PATH + 'Sample2_scored.txt')

    # print total time to run Part A
    print "Part A time: " + str(time.clock()) + ' sec'

if __name__ == "__main__": main()
