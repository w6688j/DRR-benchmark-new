import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument('--world-size', default=2, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://114.116.94.156:2222', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--dist-rank', default=0, type=int,
                    help='rank of distributed processes')

if __name__ == '__main__':
    args = parser.parse_args()
    # 初始化
    dist.init_process_group(backend=args.dist_backend,
                            init_method=args.dist_url,
                            world_size=args.world_size,
                            rank=args.dist_rank)

    CONTEXT_SIZE = 2
    EMBEDDING_DIM = 10
    # We will use Shakespeare Sonnet 2
    test_sentence = """When forty winters shall besiege thy brow,
    And dig deep trenches in thy beauty's field,
    Thy youth's proud livery so gazed on now,
    Will be a totter'd weed of small worth held:
    Then being asked, where all thy beauty lies,
    Where all the treasure of thy lusty days;
    To say, within thine own deep sunken eyes,
    Were an all-eating shame, and thriftless praise.
    How much more praise deserv'd thy beauty's use,
    If thou couldst answer 'This fair child of mine
    Shall sum my count, and make my old excuse,'
    Proving his beauty by succession thine!
    This were to be new made when thou art old,
    And see thy blood warm when thou feel'st it cold.""".split()

    trigram = [((test_sentence[i], test_sentence[i + 1]), test_sentence[i + 2])
               for i in range(len(test_sentence) - 2)]

    vocb = set(test_sentence)  # 通过set将重复的单词去掉
    word_to_idx = {word: i for i, word in enumerate(vocb)}
    idx_to_word = {word_to_idx[word]: word for word in word_to_idx}


    class NgramModel(nn.Module):
        # vocb_size 单词数 | context_size 预测单词需要的前面单词数
        def __init__(self, vocb_size, context_size, n_dim):
            super(NgramModel, self).__init__()
            self.n_word = vocb_size
            self.embedding = nn.Embedding(self.n_word, n_dim)
            self.linear1 = nn.Linear(context_size * n_dim, 128)
            self.linear2 = nn.Linear(128, self.n_word)

        def forward(self, x):
            emb = self.embedding(x)
            emb = emb.view(1, -1)
            out = self.linear1(emb)
            out = F.relu(out)
            out = self.linear2(out)
            log_prob = F.log_softmax(out, dim=1)
            return log_prob


    # 训练
    ngrammodel = NgramModel(len(word_to_idx), CONTEXT_SIZE, 100)
    ngrammodel = torch.nn.parallel.DistributedDataParallel(ngrammodel)

    criterion = nn.NLLLoss()
    optimizer = torch.optim.SGD(ngrammodel.parameters(), lr=1e-3)

    for epoch in range(10):
        print('epoch: {}'.format(epoch + 1))
        print('*' * 10)
        running_loss = 0
        for data in torch.utils.data.distributed.DistributedSampler(trigram):
            word, label = data
            word = Variable(torch.LongTensor([word_to_idx[i] for i in word]))
            label = Variable(torch.LongTensor([word_to_idx[label]]))
            # forward
            out = ngrammodel(word)
            loss = criterion(out, label)

            running_loss += loss.data.item()
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Loss: {:.6f}'.format(running_loss / len(word_to_idx)))

    # Save Models
    torch.save(ngrammodel, './models/ngrammodel.pkl')
