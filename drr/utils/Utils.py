from drr.utils.BuildDict import BuildDict
from drr.utils.DataSet import DataSet


class Utils:
    def __init__(self, opts):
        self.path = opts['path']

    def getSentencesAndDict(self):
        dict = (BuildDict({
            'path': self.path
        })).run()

        sentences = (DataSet({
            'path': self.path,
            'dict_dict': dict,
        })).getSentences()

        return (sentences, dict)
