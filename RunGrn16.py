import run

if __name__ == '__main__':
    (run.RunGrn16({
        'train_path': 'data/train.raw.txt',
        'test_path': 'data/test.raw.txt',
        'model_path': 'saved_models/Grn16Model.pkl'
    })).runTrain()
