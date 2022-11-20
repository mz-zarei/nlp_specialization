import numpy as np
import pandas as pd
import argparse
import utils
import os


parser = argparse.ArgumentParser(description='train a naive bayes classifier for tweet semantic analysis')    
      
parser.add_argument('--split-ratio', type=float, default=0.8,
                    help='train test split ratio')           
parser.add_argument('--verbos', default=True,
                    help='enables showing result in each step')

args = parser.parse_args()

# settings
split_ratio  = args.split_ratio
verbos       = args.verbos


outdir = './out'
if not os.path.exists(outdir):
    os.mkdir(outdir)

def main():
    
    # Load and prepare data set
    train_x, train_y, test_x, test_y = utils.prepare_dataset(split_ratio)
    if verbos:
        print('data loaded successfully!')

    # create frequency dictionary
    freqs = utils.build_freqs(train_x, train_y)
    if verbos:
        print('frequency dict is created successfully!')
    
    # train a NB
    loglikelihood = utils.train_naive_bayes(freqs, train_y)
    if verbos:
        print('Training is done successfully! loglikelihood dictionary is saved at ./out/nb_weights.npy')
    
    np.save('./out/nb_weights.npy', loglikelihood) 

    # test accuracy
    test_accuracy = utils.test_naive_bayes(test_x, test_y, loglikelihood)
    print(f'Test accuracy: {test_accuracy}')
    
    # error analysis
    # Some error analysis done for you
    print('\n Error Analysis:\n')
    print('Truth Predicted Tweet')
    for x, y in zip(test_x, test_y):
        y_hat = utils.naive_bayes_predict(x, loglikelihood)
        if y != (np.sign(y_hat) > 0):
            print('%d\t%0.2f\t%s' % (y, np.sign(y_hat) > 0, ' '.join(
                utils.process_tweet(x)).encode('ascii', 'ignore')))


if __name__ == '__main__':
    main()