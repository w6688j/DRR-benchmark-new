import torch
import torch.nn as nn
import torch.nn.functional as F

# 自定义
import drr.utils as DrrUtils

# gpu or not
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RNN(nn.Module):
    def __init__(self, hidden_size, out_size, n_layers=1, batch_size=1):
        super(RNN, self).__init__()

        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.out_size = out_size

        # 这里指定了 BATCH FIRST
        self.gru = torch.nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True)

        # 加了一个线性层，全连接
        self.out = torch.nn.Linear(hidden_size, out_size)

    def forward(self, word_inputs, hidden):
        # -1 是在其他确定的情况下，PyTorch 能够自动推断出来，view 函数就是在数据不变的情况下重新整理数据维度
        # batch, time_seq, input
        inputs = word_inputs.view(self.batch_size, -1, self.hidden_size)

        # hidden 就是上下文输出，output 就是 RNN 输出
        output, hidden = self.gru(inputs, hidden)

        output = self.out(output)

        # 仅仅获取 time seq 维度中的最后一个向量
        # the last of time_seq
        output = output[:, -1, :]
        # output = F.softmax(output, dim=1)

        return output, hidden

    def init_hidden(self):
        # 这个函数写在这里，有一定迷惑性，这个不是模型的一部分，是每次第一个向量没有上下文，在这里捞一个上下文，仅此而已。
        hidden = torch.autograd.Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        return hidden


def training():
    sentences, dict = (DrrUtils.Utils({
        'train_path': 'data/train.small.txt',
        'path': 'data/train.small.txt',
        'batch_size': 80
    })).getRnnAtt17SentencesAndDict()

    # 隐层 300，输出 4，隐层用词向量的宽度，输出用标签的值得个数 （one-hot)
    encoder_test = RNN(300, 4).to(device)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(encoder_test.parameters(), lr=0.001, momentum=0.9)

    for i in range(100):
        print("*" * 10 + "epoch:" + str(i))
        for data in sentences:
            sentence, label = data
            encoder_hidden = encoder_test.init_hidden().to(device)
            input_data = torch.autograd.Variable(torch.LongTensor(sentence))
            embedding = nn.Embedding(
                len(dict['word2id']),
                300,
                padding_idx=0
            )

            input_data = embedding(input_data)[0].to(device)
            output_labels = torch.autograd.Variable(torch.LongTensor([label])).to(device)

            encoder_outputs, encoder_hidden = encoder_test(input_data, encoder_hidden)
            print(encoder_outputs)
            print(output_labels)
            optimizer.zero_grad()
            loss = criterion(encoder_outputs, output_labels)
            loss.backward()
            optimizer.step()

            print("loss: ", loss.data.item())

    # 保存模型
    torch.save(encoder_test, 'saved_models/RNNModel.pkl')


def testing():
    # 预测
    sentences, dict = (DrrUtils.Utils({
        'train_path': 'data/train.small.txt',
        'path': 'data/test.small.txt',
        'batch_size': 80
    })).getRnnAtt17SentencesAndDict()

    id2label = dict['id2label']
    true_count = 0
    # 加载模型
    RNNModel = torch.load('saved_models/RNNModel.pkl').to(device)
    for data in sentences:
        sentence, label = data
        encoder_hidden = RNNModel.init_hidden().to(device)
        input_data = torch.autograd.Variable(torch.LongTensor(sentence))
        embedding = nn.Embedding(
            len(dict['word2id']),
            300,
            padding_idx=0
        )

        input_data = embedding(input_data)[0].to(device)
        encoder_outputs, encoder_hidden = RNNModel(input_data, encoder_hidden)
        # axis = 0 按列 axis = 1 按行
        _, predict_label = torch.max(encoder_outputs, 1)

        pre_label = predict_label.item()
        print(encoder_outputs)
        print(str(pre_label) + '-' + str(label))
        if (label == pre_label):
            true_count += 1
        print(id2label[pre_label] + '-' + id2label[label])

    print('正确率：{:.6f}%'.format((true_count / len(sentences)) * 100))


def _test_rnn_rand_vec():
    import random

    # 这里随机生成一个 Tensor，维度是 1000 x 10 x 200；其实就是1000个句子，每个句子里面有10个词向量，每个词向量 200 维度，其中的值符合 NORMAL 分布。

    _xs = torch.randn(1000, 10, 200)
    _ys = []

    # 标签值 0 - 3 闭区间
    for i in range(1000):
        _ys.append(random.randint(0, 3))

    # 隐层 200，输出 6，隐层用词向量的宽度，输出用标签的值得个数 （one-hot)
    encoder_test = RNN(200, 4)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(encoder_test.parameters(), lr=0.001, momentum=0.9)

    for i in range(_xs.size()[0]):
        encoder_hidden = encoder_test.init_hidden()

        input_data = torch.autograd.Variable(_xs[i])
        output_labels = torch.autograd.Variable(torch.LongTensor([_ys[i]]))

        print(output_labels)

        encoder_outputs, encoder_hidden = encoder_test(input_data, encoder_hidden)

        print(encoder_outputs)
        exit()

        optimizer.zero_grad()
        loss = criterion(encoder_outputs, output_labels)
        loss.backward()
        optimizer.step()

        print("loss: ", loss.data[0])

    return


if __name__ == '__main__':
    # _test_rnn_rand_vec()
    # training()
    testing()
