import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

# 自定义
import drr.models as DrrModels
import drr.utils as DrrUtils

# Hyper Parameters
EPOCH = 10  # 训练整批数据多少次, 为了节约时间, 我们只训练一次
BATCH_SIZE = 10  # 把数据集分批 每批10个句子
LR = 1e-3  # 学习率


class RunRnnatt17:
    def __init__(self, opts):
        self.train_path = opts['train_path']
        self.test_path = opts['test_path']
        self.model_path = opts['model_path']

    def runTrain(self):
        sentences, dict = (DrrUtils.Utils({
            'train_path': self.train_path,
            'path': self.train_path
        })).getSentencesAndDict()

        RNNAtt17Model = DrrModels.RNNAtt17({
            'vocab_size': len(dict['word2id'])
        })

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(RNNAtt17Model.parameters(), lr=LR)

        for epoch in range(EPOCH):
            print('epoch: {}'.format(epoch + 1))
            print('****************************')
            running_loss = 0
            sentenceArr = np.zeros((BATCH_SIZE, 256))
            labelArr = np.zeros((BATCH_SIZE, 1))
            step = 0
            for data in sentences:
                sentence, label = data
                labelArr[step] = np.array([label])
                sentenceArr[step] = sentence
                if ((step + 1) % BATCH_SIZE == 0):
                    sentenceArr = Variable(torch.LongTensor(sentenceArr))
                    labelArr = Variable(torch.LongTensor(labelArr))
                    # forward
                    out = RNNAtt17Model(sentenceArr)
                    loss = criterion(out, labelArr.squeeze_())
                    running_loss += loss.data.item()

                    print('loss')
                    print(loss.data.item())

                    # backward
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    sentenceArr = np.zeros((BATCH_SIZE, 256))
                    labelArr = np.zeros((BATCH_SIZE, 1))
                    step = -1
                step += 1
            print('Loss: {:.6f}'.format(running_loss / len(sentences)))

        # 保存模型
        torch.save(RNNAtt17Model, self.model_path)

    def runTest(self):
        # 预测
        sentences, dict = (DrrUtils.Utils({
            'train_path': self.train_path,
            'path': self.test_path
        })).getSentencesAndDict()

        id2label = dict['id2label']

        # 加载模型
        RNNAtt17Model = torch.load(self.model_path)

        sentenceArr = np.zeros((BATCH_SIZE, 256))
        labelArr = np.zeros((BATCH_SIZE, 1))
        step = 0
        true_count = 0
        for data in sentences:
            sentence, label = data
            labelArr[step] = np.array([label])
            sentence = np.array(sentence)
            sentence.resize((256,))
            sentenceArr[step] = sentence
            if ((step + 1) % BATCH_SIZE == 0):
                sentenceArr = Variable(torch.LongTensor(sentenceArr))
                out = RNNAtt17Model(sentenceArr)
                # axis = 0 按列 axis = 1 按行
                _, predict_label = torch.max(out, 1)
                print(label)
                print(predict_label)
                for i in predict_label.numpy():
                    if (id2label[i] == id2label[label]):
                        true_count += 1
                    print(id2label[i] + '-' + id2label[label])
                sentenceArr = np.zeros((BATCH_SIZE, 256))
                labelArr = np.zeros((BATCH_SIZE, 1))
                step = -1
            step += 1

        print('正确率：{:.6f}%'.format((true_count / len(sentences)) * 100))
