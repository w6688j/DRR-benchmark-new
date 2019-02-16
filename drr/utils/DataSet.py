import numpy as np
import torch.utils.data as Data


class DataSet:
    def __init__(self, opts):
        self.path = opts['path']
        self.batch_size = opts['batch_size']
        self.vocab_size = len(opts['dict_dict']['word2id'])
        self.word2id = opts['dict_dict']['word2id']
        self.label2id = opts['dict_dict']['label2id'] # {'Expansion': 0, 'Contingency': 1, 'Comparison': 2, 'Temporal': 3}

    def getRnnAtt17Sentences(self):
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
                    if (str(arg1Item) in self.word2id.keys()):
                        arg1WordIDList.append(self.word2id[str(arg1Item)])
                # 循环arg2句子的每个词，查出词在word2id中对应的下标,作为arg2WordIDList的元素
                for arg2Item in arg2:
                    if (str(arg2Item) in self.word2id.keys()):
                        arg2WordIDList.append(self.word2id[str(arg2Item)])
                # 将arg1WordIDList与arg2WordIDList直接拼接
                item = np.array(arg1WordIDList + arg2WordIDList).reshape((1, -1))
                item = item.copy()
                item.resize((1, 256))
                sentenceItem.append(item)
                # 将标签加入，组合成每一行的向量
                sentenceItem.append(self.label2id[label])
                # 将每行向量append进入整个数据集中形成矩阵
                sentences.append(sentenceItem)

            sentences = np.array(sentences)

            return sentences

    def getGrn16TensorDataset(self, params):
        torch_dataset = Data.TensorDataset(params['arg1List'], params['arg2List'], params['labelList'])
        loader = Data.DataLoader(
            dataset=torch_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2  # set multi-work num read data
        )

        return loader

    def getGrn16Sentences(self):
        with open(self.path, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
            arg1List = []
            arg2List = []
            labelList = []
            for line in lines:
                label, arg1, arg2 = self.formateArg(line)
                # 参数一的单词对应词向量的下标
                arg1WordIDList = []
                # 参数二的单词对应词向量的下标
                arg2WordIDList = []
                # 循环arg1句子的每个词，查出词在word2id中对应的下标,作为arg1WordIDList的元素
                for arg1Item in arg1:
                    if (str(arg1Item) in self.word2id.keys()):
                        arg1WordIDList.append(self.word2id[str(arg1Item)])
                # 循环arg2句子的每个词，查出词在word2id中对应的下标,作为arg2WordIDList的元素
                for arg2Item in arg2:
                    if (str(arg2Item) in self.word2id.keys()):
                        arg2WordIDList.append(self.word2id[str(arg2Item)])

                # 将arg1WordIDList填充成长度为50
                arg1WordIDList = self.resizeList(arg1WordIDList, 50)

                arg1List.extend(arg1WordIDList)

                # 将arg2WordIDList填充成长度为50
                arg2WordIDList = self.resizeList(arg2WordIDList, 50)
                arg2List.extend(arg2WordIDList)

                labelList.append(self.label2id[label])

            return (arg1List, arg2List, labelList)

    def formateArg(self, line):
        line_split = line.split('|||')
        label = line_split[0]
        arg1 = line_split[1].split()
        arg2 = line_split[2].split()

        return (label, arg1, arg2)

    def resizeList(self, list, dim):
        list = np.array(list).reshape((1, -1))
        list = list.copy()
        list.resize((1, dim))

        return list
