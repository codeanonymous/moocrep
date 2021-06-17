import matplotlib
import json

matplotlib.use('Agg')
import sys
from math import sqrt
from torch.utils.data import TensorDataset
import pickle as pkl
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import torch
import random
import copy
# from tensorflow.contrib.layers.python.layers import fully_connected
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from data_processing.siamese_data_train_test import WORD  # load the data and process it
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_recall_fscore_support, average_precision_score, classification_report
import sklearn.metrics as metrics
from pytorch_pretrained_bert import BertModel, BertTokenizer
np.random.seed(seed=1)
tf.compat.v1.disable_eager_execution()
random.seed(0)
tf.random.set_seed(0)

# hyper parameters
max_iter = 3500  # maximum number of iterations for training
learning_rate = 0.001
# 0.001
summaries_dir = './siamese_summary'


# Architecture of the siamese network
n_inputs = 128  # 100 # dimension of each of the input vectors
n_steps = 1  # sequence length
n_hidden = 128  # 64#128 #256 #128 # number of neurons of the bi-directional LSTM
n_classes = 1  # two possible classes, either `same` of `different`

adj = pkl.load(open('../../../gae-master/gae-master/gae/data/ind.mooc.graph_ce','rb'))
real_adj=pkl.load(open('../../../gae-master/gae-master/gae/data/ind.mooc.graph_real_ce','rb'))

# for i in f:
print("Adj shaepe", adj.shape)
num_concepts = adj.shape[0]
text = False
print("Real adj ",len(real_adj))
data_x1, data_x2, data_y=[],[],[]
fakes=0
for i in range(num_concepts):

    for j in range(num_concepts):

        if [i,j] in real_adj:
            # if int(i)==0 and int(j)==3:
            #     print("Checking")
            k=0
            while k  <num_concepts:
                if [i,k] in real_adj and [k, j] in real_adj:
                    fakes+=1
                    break
                k+=1

            if k <num_concepts:
                continue
            data_x1.append(int(i))
            data_x2.append(int(j))
            data_y.append(1)
            # print(str(i) + ',' + str(j) + ',' + str(adj[i][j]))
            n_neg=0
            while n_neg< 3:
            # for k in range(num_concepts):
                k=random.randint(0,num_concepts-1)
                if adj[i][k]==1:
                    continue
                n_neg+=1
                data_x1.append(int(i))
                data_x2.append(int(k))
                data_y.append(int(adj[i][k]))
                # print(str(i)+','+str(k)+','+str(adj[i][k]))


print(fakes)
x1 = tf.compat.v1.placeholder(tf.int32,
                              shape=[None])  # placeholder for the first network (concept 1)
x2 = tf.compat.v1.placeholder(tf.int32,
                              shape=[None])  # placeholder for the second network (concept 2)

# placeholder for the label. `1` for `same` and `0` for `different`.
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

# placeholder for dropout (we could use different dropout for different part of the architecture)
keep_prob = tf.compat.v1.placeholder(tf.float32)
train_phase = tf.compat.v1.placeholder(tf.bool)
bert = BertModel.from_pretrained('bert-base-chinese').cuda()
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

def train_tensors(texts):
    train_tokens_tensor, train_masks_tensor = text_to_train_tensors(texts, tokenizer, 50)

    # train_dataset = TensorDataset(train_tokens_tensor, train_masks_tensor)
    # train_sampler = RandomSampler(train_dataset)
    return train_tokens_tensor, train_masks_tensor


def text_to_train_tensors(texts, tokenizer, max_seq_length):

        # All features

    # if tokenizer == None:

    if tokenizer:
        train_tokens = list(map(lambda t: ['[CLS]'] + tokenizer.tokenize(t)[:max_seq_length - 1], texts))
        train_tokens_ids = list(map(tokenizer.convert_tokens_to_ids, train_tokens))
    train_tokens_ids = pad_sequences(train_tokens_ids, maxlen=max_seq_length, truncating="post", padding="post",
                                     dtype="int")

    train_masks = [[float(i > 0) for i in ii] for ii in train_tokens_ids]

    # to tensors
    # train_tokens_tensor, train_masks_tensor
    return train_tokens_ids, train_masks

def reshape_input(x_):
    """
	Reshape the inputs to match the shape requirements of the function
	`tf.nn.bidirectional_rnn`

	Args:
		x_: a tensor of shape `(batch_size, n_steps, n_inputs)`

	Returns:
		A `list` of length `n_steps` with its elements being tensors of shape `(batch_size, n_inputs)`
	"""
    x_ = tf.transpose(a=x_, perm=[1, 0, 2])  # shape: (n_steps, batch_size, n_inputs)
    x_ = tf.split(x_, n_steps, 0)  # tensor flow > 0.12
    # x_ = tf.split(0, n_steps, x_) # a list of `n_steps` tensors of shape (1, batch_size, n_steps)
    return [tf.squeeze(z, [0]) for z in x_]  # remove size 1 dimension --> (batch_size, n_steps)


def add_fc(inputs, outdim, train_phase, scope_in):
    fc = tf.compat.v1.layers.dense(inputs, outdim, activation=None,kernel_initializer=None,reuse=tf.compat.v1.AUTO_REUSE, name=scope_in + '/fc')
    fc_bnorm = tf.compat.v1.layers.batch_normalization(fc, momentum=0.1, epsilon=1e-5,
                                                       training=train_phase, name=scope_in + '/bnorm')
    fc_relu = tf.nn.relu(fc_bnorm, name=scope_in + '/relu')
    fc_out = tf.compat.v1.layers.dropout(fc_relu, rate=0.1, seed=0, training=train_phase, name=scope_in + '/dropout')
    return fc_out

def add_fc2(inputs, outdim, train_phase, scope_in):
    fc = tf.compat.v1.layers.dense(inputs, outdim, activation=None,kernel_initializer=None,reuse=tf.compat.v1.AUTO_REUSE, name=scope_in + '/fc')
    fc_bnorm = tf.compat.v1.layers.batch_normalization(fc, momentum=0.1, epsilon=1e-5,
                                                       training=train_phase, name=scope_in + '/bnorm')
    fc_relu = tf.nn.relu(fc_bnorm, name=scope_in + '/relu')
    fc_out = tf.compat.v1.layers.dropout(fc_relu, rate=0.1, seed=0, training=train_phase, name=scope_in + '/dropout')
    return fc_out


f_emb = open('../../../../anaconda3/envs/myenv/Lib/site-packages/concept_embeddings.tsv','r')
e1 = np.zeros((num_concepts,128))
# for i in range(e1.shape[0]):
#     for j in range(e1.shape[1]):
#         e1[i][j]=np.random.normal(0, 0.1)
# for cnt,i in enumerate(f_emb):
#     # if cnt in pro_id_dict.keys():
#     e1[cnt]=[float(j) for j in i.split(',')]

embed_data = torch.load('../../../../pythonProject/course_rec/embedding.pt')
e2=np.array(embed_data.detach().cpu())
for i in range(e1.shape[0]):
    e1[i]=e2[i]

pro_embed_init = tf.compat.v1.constant_initializer(e1)
# print(e2.shape)3
print("Num Cocnepts", num_concepts)
print("*****************************")
# pro_embed_init = tf.compat.v1.truncated_normal_initializer(stddev=0.1)
embs = tf.compat.v1.get_variable('embed_matrix', [num_concepts, 128 ],
                                initializer=pro_embed_init, trainable=False)

def embedding_model(feats, train_phase, scope_name,
                    fc_dim=n_inputs, embed_dim=n_hidden):
    """0.8
		Build two-branch embedding networks.
		fc_dim: the output dimension of the first fc layer.
		embed_dim: the output dimension of the second fc layer, i.e.
				   embedding space dimension.
	"""
    # each branch.
    fc1 = add_fc2(feats, fc_dim, train_phase, scope_name)
    fc2 = tf.compat.v1.layers.dense(fc1, embed_dim, activation=None,kernel_initializer=None,reuse=tf.compat.v1.AUTO_REUSE, name = scope_name + '_2')
    embed = tf.nn.l2_normalize(fc2, 1, epsilon=1e-10)
    return embed


if text == False:
    x1_, x2_ = tf.nn.embedding_lookup(params=embs, ids = x1), tf.nn.embedding_lookup(params=embs, ids = x2)

with tf.compat.v1.variable_scope('siamese_network') as scope:
    #	 with tf.name_scope('Embed_1'):
    embed_1 = embedding_model(x1_, train_phase, 'Embed')
    #	 with tf.name_scope('Embed_2'):
    reuse = True
    scope.reuse_variables()  # tied weights (reuse the weights)
    embed_2 = embedding_model(x2_, train_phase, 'Embed')

# with tf.compat.v1.variable_scope('sim_encoder_network') as scope:
#     #	 with tf.name_scope('Embed_1'):
#     embed_1 = embedding_model(x1_, train_phase, 'Embed')
#     #	 with tf.name_scope('Embed_2'):
#     reuse = True
#     scope.reuse_variables()  # tied weights (reuse the weights)
#     embed_2 = embedding_model(x2_, train_phase, 'Embed')

# Weights and biases for the layer that connects the outputs from the two networks
weights2 = tf.compat.v1.get_variable('weigths_out2', shape=[2*n_hidden, n_hidden],
                                    initializer=tf.compat.v1.random_normal_initializer(stddev=1.0 / float(n_hidden)))
biases2 = tf.compat.v1.get_variable('biases_out2', shape=[n_hidden])

weights = tf.compat.v1.get_variable('weigths_out', shape=[n_hidden, n_classes],
                                    initializer=tf.compat.v1.random_normal_initializer(stddev=1.0 / float(n_hidden)))
biases = tf.compat.v1.get_variable('biases_out', shape=[n_classes])

# last_state1 = tf.squeeze(embed_1)
# last_state2 = tf.squeeze(embed_2)
# last_states_diff = tf.squeeze(tf.abs(embed_1 - embed_2), [0])
last_states_diff = tf.concat([embed_2, embed_1], axis=1)
# last_states_diff = tf.compat.v1.Print(n_hidden,[last_states_diff] )
# last_states_diff = tf.compat.v1.Print(weights2.shape,[last_states_diff] )
last_states_diff = tf.linalg.matmul(last_states_diff, weights2) + biases2
# embed_1[0] = tf.compat.v1.Print(embed_1[0],[embed_1[0]])
# last_states_diff = embed_1 - embed_2

logits = tf.compat.v1.matmul(last_states_diff, weights) + biases

# prediction = tf.nn.log_softmax (logits=logits)
# logits = tf.compat.v1.Print(input_=logits, data=[logits])
# y = tf.compat.v1.Print(input_ = y, data=[y])
loss = tf.nn.weighted_cross_entropy_with_logits(logits=logits, labels=y, pos_weight= 5)

# optimizer
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(tf.reduce_mean(input_tensor=loss))

pred_prob = tf.nn.sigmoid(logits)

with tf.compat.v1.name_scope('total'):
    cross_entropy = tf.reduce_mean(input_tensor=loss)
tf.compat.v1.summary.scalar('cross_entropy', cross_entropy)
# tf.compat.v1.summary.scalar('accuracy', accuracy)
merged = tf.compat.v1.summary.merge_all()
train_writer = tf.compat.v1.summary.FileWriter(summaries_dir + '/train')
test_writer = tf.compat.v1.summary.FileWriter(summaries_dir + '/test')


def calculate_precision_from_logits(y):
    n = len(y)
    count = 0
    for i in range(n):
        if y[i][2] == np.argmax(y[i][3]):
            count = count + 1
    return float(count) / float(n)


def calculate_recall_from_logits(y):
    n = len(y)
    count = 0
    total = 0
    for i in range(n):
        if y[i][2] == np.argmax(y[i][3]):
            count = count + y[i][2]
        if y[i][2] == 1:
            total = total + 1
    return float(count) / float(total)


n_runs = 100

init = tf.compat.v1.global_variables_initializer()
print("# of non zeros",np.sum(data_y))
print("Total elemts", len(data_y))

data = list(zip(data_x1,data_x2, data_y))
random.shuffle(data)

n_train = int(len(data)*0.6)
n_test = int(len(data)*0.8)
data=np.array(data)
# for i in range(len(data)):
#     if data[i,0] == 0 and data[i,1]==3:
#         print("Here")

train_data, test_data  = data[:n_train,:], data[n_train:,:]
batch_size=256
best_auc,best_acc, bestf1, class_report,best_prediction = 0,0,0,0,[]
with tf.compat.v1.Session() as sess:
    sess.run(init)  # initialize all variables
    print('Network training begins.')
    for _ in range(1, n_runs + 1):
        # We retrieve a batch of data from the training set
        # batch_x1, batch_x2, batch_y = data.get_next_batch(batch_train, phase='train')
        for i in range(0, len(train_data), batch_size):
            batch_x1, batch_x2, batch_y = train_data[i:i+batch_size,0], train_data[i:i+batch_size,1], train_data[i:i+batch_size,2].reshape(-1,1)
            # We feed the data to the network for training

            feed_dict = {x1: batch_x1, x2: batch_x2, y: batch_y, keep_prob: .9, train_phase: True}

            _, loss_, summary = sess.run([optimizer, loss, merged], feed_dict=feed_dict)


            train_writer.add_summary(summary, i)

        print('********************************')
        print('Training finished.')
        # testing the trained network on a large sample
        print('********************************')
        print('Testing the network.')
        print('********************************')
        print('Number of Test samples : ', len(test_data))
        predicted = []
        for i in range(0,len(test_data), batch_size):
            batch_x1, batch_x2, batch_y =test_data[i:i + batch_size,0], test_data[i:i + batch_size,1], test_data[i:i + batch_size,2].reshape(-1,1)

            feed_dict = {x1: batch_x1, x2: batch_x2, y: batch_y, keep_prob: 1.0, train_phase: False}
            pred_prob_, loss_test = sess.run([ pred_prob, loss],   feed_dict=feed_dict)
            predicted.extend([i[0] for i in pred_prob_])
            # pred2=np.array([i[0] for i in pred_prob_])
            # # print(pred2+)
            # pred2[pred2 >= 0.8] = 1.
            # pred2[pred2 < 0.8] = 0.

            # for i in range(len(x1__)):
            #     # print(batch_y[i])
            #     if batch_y[i][0]==0.:
            #         print(str(x1__[i])+','+str(x2__[i]))
            # assert(len(batch_y) == len(pred2))
            # for j in range(len(batch_y)):
            #     if batch_y[j] !=pred2[j] and batch_y[i] == 1.:
            #         print(str(batch_x1[j])+',' +str(batch_x2[j]))

        print(metrics.roc_auc_score(test_data[:, 2], predicted))
        if metrics.roc_auc_score(test_data[:, 2], predicted) > best_auc:
            best_auc = metrics.roc_auc_score(test_data[:, 2], predicted)
            predicted = np.array(predicted)

            fpr, tpr, thresholds = roc_curve(test_data[:, 2], predicted)
            gmeans = np.array([sqrt(tpr[i] * (1 - fpr[i])) for i in range(len(tpr))])
            ix = np.argmax(gmeans)
            print("threshold",thresholds[ix])
            pred2=copy.deepcopy(predicted)

            for it in np.arange(0.,1.,0.001):

                predicted[pred2 >= it] = 1.
                predicted[pred2 < it] = 0.


                prec_recall_fscore = precision_recall_fscore_support(test_data[:, 2], predicted, average='macro')

                if prec_recall_fscore[2]<bestf1:
                    continue
                bestf1 = prec_recall_fscore[2]
                best_prec_recall = prec_recall_fscore
                best_acc = accuracy_score(test_data[:, 2], predicted)

                class_report = classification_report(test_data[:, 2], predicted)
        # print(best_auc)
    print("AUC", best_auc)
    print(best_prec_recall)
    print("ACC", best_acc)
    print(class_report)
