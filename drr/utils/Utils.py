from drr.utils.BuildDict import BuildDict
from drr.utils.DataSet import DataSet


class Utils:
    def __init__(self, opts):
        self.train_path = opts['train_path']
        self.path = opts['path']

    def getSentencesAndDict(self):
        dict = (BuildDict({
            'path': self.train_path
        })).run()

        sentences = (DataSet({
            'path': self.path,
            'dict_dict': dict,
        })).getSentences()

        return (sentences, dict)
