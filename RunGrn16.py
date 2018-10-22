import run

if __name__ == '__main__':
    (run.RunGrn16({
        'train_path': 'data/train.small.txt',
        'test_path': 'data/test.small.txt',
        'model_path': 'saved_models/Grn16ModelSmall.pkl'
    })).runTest()
