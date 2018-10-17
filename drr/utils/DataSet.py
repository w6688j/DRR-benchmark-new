import numpy as np


class DataSet:
    def __init__(self, opts):
        self.path = opts['path']
        self.vocab_size = len(opts['dict_dict']['word2id'])
        self.word2id = opts['dict_dict']['word2id']
        self.label2id = opts['dict_dict']['label2id']

    def getSentences(self):
        with open(self.path, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
            sentences = []
            for line in lines:
                label, arg1, arg2 = self.formateArg(line)
                # 每句话对应的词向量下标组成的向量,最后一列为标签下标
                sentenceItem = []
                # 参数一的单词对应词向量的下标
                arg1WordIDList = []
                # 参数二的单词对应词向量的下标
                arg2WordIDList = []
                # 循环arg1句子的每个词，查出词在word2id中对应的下标,作为arg1WordIDList的元素
                for arg1Item in arg1:
                    arg1WordIDList.append(self.word2id[str(arg1Item)])
                # 循环arg2句子的每个词，查出词在word2id中对应的下标,作为arg2WordIDList的元素
                for arg2Item in arg2:
                    arg2WordIDList.append(self.word2id[str(arg2Item)])
                # 将arg1WordIDList与arg2WordIDList直接拼接
                sentenceItem.append(arg1WordIDList + arg2WordIDList)
                # 将标签加入，组合成每一行的向量
                sentenceItem.append(self.label2id[label])
                # 将每行向量append进入整个数据集中形成矩阵
                sentences.append(sentenceItem)

            sentences = np.array(sentences)

            return sentences

    def formateArg(self, line):
        line_split = line.split('|||')
        label = line_split[0]
        arg1 = line_split[1].split()
        arg2 = line_split[2].split()

        return (label, arg1, arg2)
