import re
import string
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from os import getcwd
from nltk.corpus import twitter_samples 


def prepare_dataset(train_test_ratio = 0.8):
    '''Download and train/test split the tweet data set from nltk
    Input:
        train_test_ratio: ratio of train test split
    Output:
        train_x, train_y: train data 
        test_x, test_y: test data
    '''

    # download tweet data set
    nltk.download('twitter_samples', download_dir="./")
    nltk.download('stopwords', download_dir="./")

    # add path to data folder to nltk path list
    filePath = f"{getcwd()}/"
    nltk.data.path.append(filePath)

    # select the set of positive and negative tweets
    all_positive_tweets = twitter_samples.strings('positive_tweets.json')
    all_negative_tweets = twitter_samples.strings('negative_tweets.json')

    # split the data into two pieces, one for training and one for testing (validation set) 
    ind = int(len(twitter_samples.strings('positive_tweets.json')) * train_test_ratio)
    test_pos = all_positive_tweets[ind:]
    train_pos = all_positive_tweets[:ind]
    test_neg = all_negative_tweets[ind:]
    train_neg = all_negative_tweets[:ind]

    train_x = train_pos + train_neg 
    test_x = test_pos + test_neg

    # combine positive and negative labels
    train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)))
    test_y = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)))
    
    return train_x, train_y, test_x, test_y

def process_tweet(tweet):
    """Preprocess a given tweet .
    Input:
        tweet: a string containing a tweet
    Output:
        tweets_clean: a list of words containing the processed tweet

    """
    

    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks    
    tweet = re.sub(r'https?://[^\s\n\r]+', '', tweet)
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweet_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
            # tweets_clean.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            tweet_clean.append(stem_word)

    return tweet_clean

def build_freqs(tweets, ys):
    """Build frequencies map.
    Input:
        tweets: a list of tweets
        ys: an m x 1 array with the sentiment label of each tweet
            (either 0 or 1)
    Output:
        freqs: a dictionary mapping each (word, sentiment) pair to its
        frequency
    """

    # Start with an empty dictionary and populate it by looping over all tweets
    # and over all processed words in each tweet.
    freqs = {}
    for y, tweet in zip(ys, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            freqs[pair] = freqs.get(pair, 0) + 1

    return freqs

def train_naive_bayes(freqs, train_y):
    '''train a naive bayes classifier

    Input:
        freqs: dictionary from (word, label) to how often the word appears
        train_x: a list of tweets
        train_y: a list of labels correponding to the tweets (0,1)
    Output:
        loglikelihood: a dictionary of the log likelihood: log(P(w_pos)/P(w_neg)) for each words, and the log prior; log(D_pos/D_neg)
    '''
    loglikelihood = {}
    logprior = 0


    # calculate V, the number of unique words in the vocabulary
    vocab = set([pair[0] for pair in freqs.keys()])
    V = len(vocab)  

    # calculate N_pos, N_neg, V_pos, V_neg
    N_pos = N_neg = 0
    for pair in freqs.keys():
        # if the label is positive (greater than zero)
        if pair[1] > 0:

            # Increment the number of positive words by the count for this (word, label) pair
            N_pos += freqs[pair]

        # else, the label is negative
        else:

            # increment the number of negative words by the count for this (word,label) pair
            N_neg += freqs[pair]
    

    # Calculate D_pos, the number of positive documents
    D_pos = (train_y == 1).sum()

    # Calculate D_neg, the number of negative documents
    D_neg = (train_y == 0).sum()

    # Calculate logprior
    logprior = np.log(D_pos/D_neg)
    
    # For each word in the vocabulary...
    for word in vocab:
        # get the positive and negative frequency of the word
        freq_pos = freqs.get((word, 1),0)
        freq_neg = freqs.get((word, 0),0)

        # calculate the probability that each word is positive, and negative
        p_w_pos = (freq_pos+1)/ (N_pos + V)
        p_w_neg = (freq_neg+1)/ (N_neg + V)

        # calculate the log likelihood of the word
        loglikelihood[word] = np.log(p_w_pos/p_w_neg)

    loglikelihood['logprior'] = logprior
    return loglikelihood

def naive_bayes_predict(tweet, loglikelihood):
    '''returns the probability that the tweet belongs to the positive or negative class
    
    Input:
        tweet: a string
        loglikelihood: a dictionary of words mapping to numbers
    Output:
        p: the sum of all the logliklihoods of each word in the tweet (if found in the dictionary) + logprior (a number)

    '''

    # process the tweet to get a list of words
    word_l = process_tweet(tweet)

    # initialize probability to zero
    p = 0

    # add the logprior
    p += loglikelihood['logprior']

    for word in word_l:

        # check if the word exists in the loglikelihood dictionary
        if word in loglikelihood:
            # add the log likelihood of that word to the probability
            p += loglikelihood[word]

    return p

def test_naive_bayes(test_x, test_y, loglikelihood):
    """ takes in your `test_x`, `test_y`, log_prior, and loglikelihood, It returns the accuracy of your model
    
    Input:
        test_x: A list of tweets
        test_y: the corresponding labels for the list of tweets
        loglikelihood: a dictionary with the loglikelihoods for each word
    Output:
        accuracy: (# of tweets classified correctly)/(total # of tweets)
    """
    accuracy = 0  # return this properly

    y_hats = []
    for tweet in test_x:
        # if the prediction is > 0
        if naive_bayes_predict(tweet, loglikelihood) > 0:
            # the predicted class is 1
            y_hat_i = 1
        else:
            # otherwise the predicted class is 0
            y_hat_i = 0

        # append the predicted class to the list y_hats
        y_hats.append(y_hat_i)

    # Accuracy is 1 minus the error
    accuracy = (y_hats == test_y).mean()

    return accuracy