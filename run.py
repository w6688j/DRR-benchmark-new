import run

if __name__ == '__main__':
    (run.RunRnnatt17({
        'train_path': 'data/train.small.txt',
        'test_path': 'data/test.small.txt',
        'model_path': 'saved_models/RNNAtt17ModelSmall.pkl'
    })).runTrain()
