import torch
import torch.nn as nn
from torch.autograd import Variable

# 自定义
import drr.models as DrrModels
import drr.utils as DrrUtils

# Hyper Parameters
import run

EPOCH = 500  # 训练整批数据多少次, 为了节约时间, 我们只训练100次
BATCH_SIZE = 32  # how many samples per batch to load
LR = 1e-3  # 学习率


class RunGrn16:
    def __init__(self, opts):
        self.train_path = opts['train_path']
        self.test_path = opts['test_path']
        self.model_path = opts['model_path']

    def runTrain(self):
        torch.manual_seed(1)

        loader, dict = (DrrUtils.Utils({
            'train_path': self.train_path,
            'path': self.train_path,
            'batch_size': BATCH_SIZE
        })).getGrn16SentencesAndDict()

        Grn16Model = DrrModels.GRN16({
            'vocab_size': len(dict['word2id']),
            'r': 2
        }).cuda()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(Grn16Model.parameters(), lr=LR)

        for epoch in range(EPOCH):
            print('epoch: {}'.format(epoch + 1))
            print('****************************')
            num = 0
            running_loss = 0

            for step, (arg1List, arg2List, labelList) in enumerate(loader):
                arg1 = Variable(arg1List.long()).cuda()
                arg2 = Variable(arg2List.long()).cuda()
                label = Variable(labelList.long()).cuda()

                # forward
                out = Grn16Model((arg1, arg2))
                loss = criterion(out, label)
                running_loss += loss.data.item()

                print('loss')
                print(loss.data.item())

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                num += loader.__len__()

            print('Loss: {:.6f}'.format(running_loss / num))

        # 保存模型
        torch.save(Grn16Model, self.model_path)

    def runTest(self):
        # 预测
        loader, dict = (DrrUtils.Utils({
            'train_path': self.train_path,
            'path': self.test_path,
            'batch_size': BATCH_SIZE
        })).getGrn16SentencesAndDict()

        id2label = dict['id2label']

        Grn16Model = torch.load(self.model_path).cuda()

        num = 0
        true_count = 0
        for step, (arg1List, arg2List, labelList) in enumerate(loader):
            arg1 = Variable(arg1List.long()).cuda()
            arg2 = Variable(arg2List.long()).cuda()
            labelList = labelList.numpy()

            out = Grn16Model((arg1, arg2))
            # axis = 0 按列 axis = 1 按行
            _, predict_label = torch.max(out, 1)
            for i in predict_label.cpu().numpy():
                if (id2label[i] == id2label[labelList[i]]):
                    true_count += 1
                print(id2label[i] + '-' + id2label[labelList[i]])

            num += loader.__len__()

        print('正确率：{:.6f}%'.format((true_count / num) * 100))


if __name__ == '__main__':
    (run.RunGrn16({
        'train_path': 'E:/Projects/PyCharmProjects/GitHub/DRR-benchmark-new/data/train.raw.txt',
        'test_path': 'E:/Projects/PyCharmProjects/GitHub/DRR-benchmark-new/data/test.raw.txt',
        'model_path': 'E:/Projects/PyCharmProjects/GitHub/DRR-benchmark-new/saved_models/Grn16Model.pkl'
    })).runTest()
