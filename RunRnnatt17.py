import sys
import run

if __name__ == '__main__':
    # action = sys.argv[1]  # Choose action from cmd line. Options: train or test
    action = 'train'
    # action = 'pre'

    if action == 'train':
        (run.RunRnnatt17({
            'train_path': 'data/train.raw.txt',
            'test_path': 'data/test.raw.txt',
            'model_path': 'saved_models/RNNAtt17Model.pkl'
        })).runTrain()
    else:
        (run.RunRnnatt17({
            'train_path': 'data/train.raw.txt',
            'test_path': 'data/test.raw.txt',
            'model_path': 'saved_models/RNNAtt17Model.pkl'
        })).runTest()
