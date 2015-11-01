import sys
import nltk
import math
import time

START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
RARE_SYMBOL = '_RARE_'
RARE_WORD_MAX_FREQ = 5
LOG_PROB_OF_ZERO = -1000


# TODO: IMPLEMENT THIS FUNCTION
# Receives a list of tagged sentences and processes each sentence to generate a list of words and a list of tags.
# Each sentence is a string of space separated "WORD/TAG" tokens, with a newline character in the end.
# Remember to include start and stop symbols in yout returned lists, as defined by the constants START_SYMBOL and STOP_SYMBOL.
# brown_words (the list of words) should be a list where every element is a list of the tags of a particular sentence.
# brown_tags (the list of tags) should be a list where every element is a list of the tags of a particular sentence.
def split_wordtags(brown_train):
    brown_words = []
    brown_tags = []

    for data in brown_train:
	words = list()
	words.append(START_SYMBOL)
	words.append(START_SYMBOL)
	tags = list()
	tags.append(START_SYMBOL)
	tags.append(START_SYMBOL)
	tokens = data.split()
	for token in tokens:
		words.append(token[:len(token)-1-token[::-1].index('/')])
		tags.append(token[len(token)-token[::-1].index('/'):])
	words.append(STOP_SYMBOL)
	tags.append(STOP_SYMBOL)
	brown_words.append(words)
	brown_tags.append(tags)

    return brown_words, brown_tags


# TODO: IMPLEMENT THIS FUNCTION
# This function takes tags from the training data and calculates tag trigram probabilities.
# It returns a python dictionary where the keys are tuples that represent the tag trigram, and the values are the log probability of that trigram
def calc_trigrams(brown_tags):
    q_values = {}
    bigram_count = {}
    trigram_count = {}
    tags_trigram = []
    tags_bigram = []
    for tag_list in brown_tags:
	tags_trigram.append(list(nltk.trigrams(tag_list)))
	tags_bigram.append(list(nltk.bigrams(tag_list[1:])))
    for trigram in tags_trigram:
	for tri in trigram:
	    if q_values.has_key(tri):
	        q_values[tri] += 1
	    else:
	        q_values[tri] = 1
    for bigram in tags_bigram:
	for bi in bigram:
	    if bigram_count.has_key(bi):
	        bigram_count[bi] += 1
	    else:
	        bigram_count[bi] = 1

    for trigram in set(q_values.keys()):
	if trigram[0] == START_SYMBOL and trigram[1] == START_SYMBOL:
	    q_values[trigram] = math.log(float(q_values[trigram])/(len(brown_tags)),2)
	else:
	    q_values[trigram] = math.log(float(q_values[trigram])/bigram_count[(trigram[0],trigram[1])],2)

    return q_values

# This function takes output from calc_trigrams() and outputs it in the proper format
def q2_output(q_values, filename):
    outfile = open(filename, "w")
    trigrams = q_values.keys()
    trigrams.sort()  
    for trigram in trigrams:
        output = " ".join(['TRIGRAM', trigram[0], trigram[1], trigram[2], str(q_values[trigram])])
        outfile.write(output + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and returns a set of all of the words that occur more than 5 times (use RARE_WORD_MAX_FREQ)
# brown_words is a python list where every element is a python list of the words of a particular sentence.
# Note: words that appear exactly 5 times should be considered rare!
def calc_known(brown_words):
    known_words = set([])
    words_count = {}
    for sentence in brown_words:
	for word in sentence:
	    if word in words_count:
		words_count[word] += 1
	    else:
		words_count[word] = 1
    for k in words_count:
	if words_count[k] > RARE_WORD_MAX_FREQ:
	    known_words.add(k)
    return set(known_words)

# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and a set of words that should not be replaced for '_RARE_'
# Returns the equivalent to brown_words but replacing the unknown words by '_RARE_' (use RARE_SYMBOL constant)
def replace_rare(brown_words, known_words):
    brown_words_rare = []
    for sentence in brown_words:
	new_sentence = []
	for word in sentence:
	    if word in known_words:
		new_sentence.append(word)
	    else:
		new_sentence.append(RARE_SYMBOL)
	brown_words_rare.append(new_sentence)
    return brown_words_rare

# This function takes the ouput from replace_rare and outputs it to a file
def q3_output(rare, filename):
    outfile = open(filename, 'w')
    for sentence in rare:
        outfile.write(' '.join(sentence[2:-1]) + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates emission probabilities and creates a set of all possible tags
# The first return value is a python dictionary where each key is a tuple in which the first element is a word
# and the second is a tag, and the value is the log probability of the emission of the word given the tag
# The second return value is a set of all possible tags for this data set
def calc_emission(brown_words_rare, brown_tags):
    e_values = {}
    taglist = set([])


    tag_count = {}
    word_tag = {}
    for i in xrange(len(brown_tags)):
	sentence = brown_words_rare[i]
	tags = brown_tags[i]
	for j in xrange(len(tags)):
	    word = sentence[j]
	    tag = tags[j]
	    if (word, tag) in word_tag:
		word_tag[(word, tag)] += 1
	    else:
	        word_tag[(word, tag)] = 1
	    if tag in tag_count:
		tag_count[tag] += 1
	    else:
		tag_count[tag] = 1

    for wt in word_tag:
	e_values[wt] = math.log(float(word_tag[wt])/tag_count[wt[1]],2)

    for tag in tag_count:
	taglist.add(tag)

    return e_values, set(taglist)

# This function takes the output from calc_emissions() and outputs it
def q4_output(e_values, filename):
    outfile = open(filename, "w")
    emissions = e_values.keys()
    emissions.sort()  
    for item in emissions:
        output = " ".join([item[0], item[1], str(e_values[item])])
        outfile.write(output + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# This function takes data to tag (brown_dev_words), a set of all possible tags (taglist), a set of all known words (known_words),
# trigram probabilities (q_values) and emission probabilities (e_values) and outputs a list where every element is a tagged sentence 
# (in the WORD/TAG format, separated by spaces and with a newline in the end, just like our input tagged data)
# brown_dev_words is a python list where every element is a python list of the words of a particular sentence.
# taglist is a set of all possible tags
# known_words is a set of all known words
# q_values is from the return of calc_trigrams()
# e_values is from the return of calc_emissions()
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a 
# terminal newline, not a list of tokens. Remember also that the output should not contain the "_RARE_" symbol, but rather the
# original words of the sentence!
def viterbi(brown_dev_words, taglist, known_words, q_values, e_values):
    tagged = []
    
    print len(brown_dev_words)
    
    pi_ini = {}
    for u in taglist:
	for v in taglist:
	    pi_ini[(u,v)] = LOG_PROB_OF_ZERO

    for item in brown_dev_words:
	
	sentence = item + [STOP_SYMBOL]
	n = len(sentence)

	cur_length = 0
	pre_path = {}
	pre_path[(START_SYMBOL,START_SYMBOL)] = [START_SYMBOL, START_SYMBOL]
        pre_bigram  = [(START_SYMBOL, START_SYMBOL)]

	pre_pi = {}
	pre_pi[(START_SYMBOL, START_SYMBOL)] = 0

	while cur_length < len(sentence):
	    cur_path = {}
	    cur_bigram = []
	    cur_pi = {}

	    if cur_length == len(sentence)-1:
		word = STOP_SYMBOL
		tag_space = [STOP_SYMBOL]
	    else:
		word = sentence[cur_length]
		if word not in known_words:
		    word = RARE_SYMBOL
		tag_space = list(taglist)

	    for v in tag_space:
		e_tmp = (word,v)
		if e_tmp not in e_values:
		    continue
		for u in taglist:
		    w_tmp = ''
		    for w in taglist:
			if (w,u) not in pre_bigram:
			    continue
			cur_trigram = (w,u,v)
			if cur_trigram not in q_values:
			   q_values[cur_trigram] = LOG_PROB_OF_ZERO
			if (u,v) not in cur_pi:
			   cur_pi[(u,v)] = pre_pi[(w,u)] + q_values[cur_trigram] + e_values[e_tmp]
			   w_tmp = w
			elif pre_pi[(w,u)] + q_values[cur_trigram] + e_values[e_tmp] > cur_pi[(u,v)]:
				cur_pi[(u,v)] = pre_pi[(w,u)] + q_values[cur_trigram] + e_values[e_tmp]
				w_tmp = w
		    if w_tmp != '':
			cur_path[(u,v)] = pre_path[(w_tmp,u)] + [v]
			cur_bigram.append((u,v))

	    
	    pre_pi = dict(cur_pi)
	    pre_bigram = list(cur_bigram)
	    pre_path = dict(cur_path)
	    cur_length += 1
	
	st = ''
	bi_max = pre_bigram[0]
	for bi in pre_bigram:
	    if cur_pi[bi] > cur_pi[bi_max]:
		bi = bi_max
	for i,tag in enumerate(cur_path[bi_max][2:-1]):
	    st = st + sentence[i] + '/' + tag + ' '
	tagged.append(st.strip()+'\n')
	
    return tagged

# This function takes the output of viterbi() and outputs it to file
def q5_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# This function uses nltk to create the taggers described in question 6
# brown_words and brown_tags is the data to be used in training
# brown_dev_words is the data that should be tagged
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a 
# terminal newline, not a list of tokens. 
def nltk_tagger(brown_words, brown_tags, brown_dev_words):
    # Hint: use the following line to format data to what NLTK expects for training
    training = [ zip(brown_words[i],brown_tags[i]) for i in xrange(len(brown_words)) ]

    # IMPLEMENT THE REST OF THE FUNCTION HERE
    tagged = []
    tag1 = nltk.DefaultTagger('NOUN')
    tag2 = nltk.BigramTagger(training, backoff = tag1)
    tag3 = nltk.TrigramTagger(training, backoff = tag2)

    for sentence in brown_dev_words:
	flag = ''
        tags = tag3.tag(sentence)
	for t in tags:
	    flag = flag + t[0]+'/'+t[1]+' '
	tagged.append(flag.strip()+'\n')
    return tagged

# This function takes the output of nltk_tagger() and outputs it to file
def q6_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'

def main():
    # start timer
    time.clock()

    # open Brown training data
    infile = open(DATA_PATH + "Brown_tagged_train.txt", "r")
    brown_train = infile.readlines()
    infile.close()

    # split words and tags, and add start and stop symbols (question 1)
    brown_words, brown_tags = split_wordtags(brown_train)

    # calculate tag trigram probabilities (question 2)
    q_values = calc_trigrams(brown_tags)

    # question 2 output
    q2_output(q_values, OUTPUT_PATH + 'B2.txt')

    # calculate list of words with count > 5 (question 3)
    known_words = calc_known(brown_words)

    # get a version of brown_words with rare words replace with '_RARE_' (question 3)
    brown_words_rare = replace_rare(brown_words, known_words)

    # question 3 output
    q3_output(brown_words_rare, OUTPUT_PATH + "B3.txt")

    # calculate emission probabilities (question 4)
    e_values, taglist = calc_emission(brown_words_rare, brown_tags)

    # question 4 output
    q4_output(e_values, OUTPUT_PATH + "B4.txt")

    # delete unneceessary data
    del brown_train
    del brown_words_rare

    # open Brown development data (question 5)
    infile = open(DATA_PATH + "Brown_dev.txt", "r")
    brown_dev = infile.readlines()
    infile.close()

    # format Brown development data here
    brown_dev_words = []
    for sentence in brown_dev:
        brown_dev_words.append(sentence.split(" ")[:-1])

    # do viterbi on brown_dev_words (question 5)
    viterbi_tagged = viterbi(brown_dev_words, taglist, known_words, q_values, e_values)

    # question 5 output
    q5_output(viterbi_tagged, OUTPUT_PATH + 'B5.txt')

    # do nltk tagging here
    nltk_tagged = nltk_tagger(brown_words, brown_tags, brown_dev_words)

    # question 6 output
    q6_output(nltk_tagged, OUTPUT_PATH + 'B6.txt')

    # print total time to run Part B
    print "Part B time: " + str(time.clock()) + ' sec'

if __name__ == "__main__": main()
