import numpy as np
import pandas as pd
import argparse
import utils
import os


parser = argparse.ArgumentParser(description='train a logistic regression for tweet semantic analysis')    
      
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

    # collect the features 'x' and stack them into a matrix 'X'
    X = np.zeros((len(train_x), 3))
    for i in range(len(train_x)):
        X[i, :]= utils.extract_features(train_x[i], freqs)

    # training labels corresponding to X
    Y = train_y

    # save features wit labels
    df = pd.DataFrame(X, columns = ['bias','positiveness', 'negativeness'])
    df['label'] = Y
    df.to_csv('./out/preccessed_data.csv', index=False)

    # Apply gradient descent
    J, theta = utils.gradientDescent(X, Y, np.zeros((3, 1)), 1e-9, 1500)

    # save theta in a text file
    np.savetxt("./out/logistic_weights.csv", theta, delimiter=",")
    if verbos:
        print('Training is done successfully! weights are saved at ./logistic_weights.csv')

    print(f"\n finall loss: {J}")
    print(f"\n theta vector (bias, positive values, negative values): {theta}") 

    test_accuracy = utils.test_logistic_regression(test_x, test_y, freqs, theta)
    print(f'Test accuracy: {test_accuracy}')



if __name__ == '__main__':
    main()