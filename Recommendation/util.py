import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue

# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample():

        user = random.choice(list(user_train.keys()))
        # while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        for i in reversed(user_train[user]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        return (user, seq, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()
# train/val/test data generation

def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(',')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)
    tot, n_train, n_valid = len(User), int(0.2* len(User)), int(0.9* len(User))
    for cnt, user in enumerate(User.keys()):
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]

        elif cnt<n_train:
            user_train[user] = User[user]
        elif cnt<n_valid:
            user_valid[user] = User[user]
        else:
            user_test[user] = User[user]
    return [user_train, user_valid, user_test, usernum, itemnum]


# TODO: merge evaluate functions for test and val set
# evaluate on test set
def evaluate(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG_10, NDCG_5, NDCG_50, NDCG_100 = 0.0, 0.0, 0.0,0.0
    test_user = 0.0
    HT_10, HT_5, HT_50, HT_100 = 0.0,0.0,0.0,0.0
    mrr =0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = test.keys()
    tot=0
    for u in users:
    #     if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen+1], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(test[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(test[u])
        # all_items = set(np.arange(1, itemnum+1))

        rated.add(0)
        predictions, items=[],[]
        # predictions.append(np.array(model.predict(*[np.array(l) for l in [[u], [seq[:-1]], [seq[1:]]]]).detach().cpu()[0]))
        items.append(seq[1:])
        for _ in range(100):

            item_idx = []
            for i in range(len(seq)-1):
                t = np.random.randint(1, itemnum + 1)
                while t in rated: t = np.random.randint(1, itemnum + 1)
                item_idx.append(t)
            # items.append(item_idx)
            item_idx = np.array(item_idx).reshape(-1)
            items.append(item_idx)
        # predictions.extend(np.array(model.predict(np.repeat([[u]], 101, axis=0), np.repeat([seq[:-1]],101,axis=0), np.array(items)).detach().cpu()))
        predictions = model.predict(u, [seq[:-1]], items)
        items = np.array(items)
        # tf.reduce_sum(
        #     ((tf.sign(self.pos_logits - self.neg_logits) + 1) / 2) * istarget
        # ) / tf.reduce_sum(istarget)
        predictions = predictions[0]
        # print(predictions)
        for cnt in range(predictions.shape[0]):
            # p = predictions[:,cnt]
            p=predictions[cnt,:]
            item_idx = torch.LongTensor(items[:,cnt])
            rank = torch.nn.PairwiseDistance()(p.unsqueeze(0).repeat(101,1), model.item_emb(item_idx)).detach().cpu()
            # print(p)
            if seq[cnt]==0:
                continue
            # rank = p.argsort().argsort()[0]
            rank = rank.argsort().argsort()[0]

        # test_user += 1

            if rank < 10:
                NDCG_10 += 1 / np.log2(rank + 2)
                HT_10 += 1
            if rank < 5:
                NDCG_5 += 1 / np.log2(rank + 2)
                HT_5 += 1
            if rank < 50:
                NDCG_50 += 1 / np.log2(rank + 2)
                HT_50 += 1
            if rank < 100:
                NDCG_100 += 1 / np.log2(rank + 2)
                HT_100 += 1
            mrr += 1/(rank+1)
            tot+=1

    return  NDCG_5 / tot, HT_5 / tot, NDCG_10 / tot, HT_10 / tot,  NDCG_50 / tot, HT_50 / tot,  NDCG_100 / tot, HT_100 / tot, mrr/tot
# evaluate on val set
def evaluate_valid(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user