import sys
import run

if __name__ == '__main__':
    action = sys.argv[1]  # Choose action from cmd line. Options: train or test

    if action == 'train':
        (run.RunGrn16({
            'train_path': 'data/train.raw.txt',
            'test_path': 'data/test.raw.txt',
            'model_path': 'saved_models/Grn16Model.pkl'
        })).runTrain()
    else:
        (run.RunGrn16({
            'train_path': 'data/train.raw.txt',
            'test_path': 'data/test.raw.txt',
            'model_path': 'saved_models/Grn16Model.pkl'
        })).runTest()
