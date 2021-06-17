import numpy as np
import torch


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num

        self.dev = args.device

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        # embed_data = np.load('../../Downloads/PEBG-master/ednet/embedding_200.npz')
        # e2, embs, e3 = embed_data['pro_repre'], embed_data['skill_repre'], embed_data['pro_final_repre']
        # embs = torch.Tensor(embs)
        #
        f_emb = open('../../anaconda3/envs/myenv/Lib/site-packages/entity_embeddings.tsv', 'r')

        e1 = np.zeros((item_num+1, 768))

        for i in range(e1.shape[0]):
            for j in range(e1.shape[1]):
                e1[i][j] = np.random.normal(0, 0.1)



        for cnt, i in enumerate(f_emb):
            # if cnt in skill_id_dict.keys():
            if cnt >= e1.shape[0]:
                break
            e1[cnt] = [float(j) for j in i.split(',')]

        # e2 = np.array(torch.load('video_embedding.pt').detach().cpu())
        # for cnt, i in enumerate(e2):
        #
        #     if cnt >= e1.shape[0]:
        #         break
        #     e1[cnt,:i.shape[0]] = i
        e1 = torch.Tensor(e1)
        self.Linear = torch.nn.Linear(768, args.hidden_units)


        embs = self.Linear(e1)


        # print("Number of videos", item_num)
        # print("Shape of emb", embs.shape)

        # for cnt, i in embs:
        #    if cnt in skill_id_dict.keys():
        #         e1[skill_id_dict[cnt]] = i
        # print(self.item_num)


        # embs = torch.nn.init.normal_(embs, mean=0.0, std=1.0)
        self.item_emb = torch.nn.Embedding.from_pretrained(embs)
        self.item_linear = torch.nn.Linear(128,128)
        # self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        # self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        #
        # self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            # new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            # self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)
            #
            # new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            # self.forward_layernorms.append(new_fwd_layernorm)
            #
            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

        self.pos_sigmoid = torch.nn.Sigmoid()
        self.rnn = torch.nn.RNN(128,128,2)
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def log2feats(self, log_seqs, pos):
        seqs = self.item_linear(self.item_emb(torch.LongTensor(log_seqs).to(self.dev)))
        # seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        # seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        # seqs = self.emb_dropout(seqs)
        pos = self.item_emb(torch.LongTensor(pos)).to(self.dev)


        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        # seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):

            seqs = torch.transpose(seqs, 0, 1)
            Q = torch.transpose(pos,0,1)
            # Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs,
                                            attn_mask=attention_mask)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            seqs = mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            # seqs = self.forward_layernorms[i](seqs)
            # seqs = self.forward_layers[i](seqs)
            # seqs *=  ~timeline_mask.unsqueeze(-1)

        # log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return seqs

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs): # for training
        # pos_feats = self.log2feats(log_seqs, pos_seqs) # user_ids hasn't been used yet
        # neg_feats = self.log2feats(log_seqs, neg_seqs)
        # pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        # neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs= seqs.transpose(0,1)
        seqs,_ = self.rnn(seqs)
        seqs = seqs.transpose(0,1)


        # pos_logits = (pos_feats).sum(dim=-1)
        # neg_logits = (neg_feats).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits+1e-32)
        # neg_pred = self.pos_sigmoid(neg_logits+1e-32)

        return seqs, self.item_emb(torch.LongTensor(pos_seqs).to(self.dev)) # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, item_indices): # for inference
        # log_feats = self.log2feats(log_seqs, item_indices) # user_ids hasn't been used yet

        # final_feat = log_feats # only use last QKV classifier, a waste
        #
        # item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)
        #
        # logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        # logits = (log_feats).sum(dim=-1)
        # preds = self.pos_sigmoid(logits) # rank same item list for different users
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs = seqs.transpose(0, 1)
        seqs, _ = self.rnn(seqs)
        seqs = seqs.transpose(0, 1)

        return seqs # preds # (U, I)