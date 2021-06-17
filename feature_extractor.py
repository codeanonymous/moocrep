import numpy as np
from collections import defaultdict
import torch
import random
u2id,c2id,id2c,c2name ={},{},{},{}
u2c= defaultdict(list)
u_cnt,i_cnt=0,0
from torch.utils.tensorboard import SummaryWriter
import jieba
writer = SummaryWriter()
from scipy import  sparse
from pytorch_pretrained_bert import BertModel
from pytorch_pretrained_bert import BertTokenizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import json
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import course_model
from elmoformanylangs import Embedder
from simple_elmo import ElmoModel

# model = ElmoModel()
torch.manual_seed(0)
random.seed(0)

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
batch_size=16



def text_to_train_tensors(texts, tokenizer, max_seq_length):

        # All features

    # if tokenizer == None:

    if tokenizer:
        train_tokens = list(map(lambda t: ['[CLS]'] + tokenizer.tokenize(t)[:max_seq_length - 1], texts))
        train_tokens_ids = list(map(tokenizer.convert_tokens_to_ids, train_tokens))
    train_tokens_ids = pad_sequences(train_tokens_ids, maxlen=max_seq_length, truncating="post", padding="post",
                                     dtype="int")

    train_masks = [[float(i > 0) for i in ii] for ii in train_tokens_ids]
    return torch.tensor(train_tokens_ids), torch.tensor(train_masks)

def train_tensors(texts):
    train_tokens_tensor, train_masks_tensor = text_to_train_tensors(texts, tokenizer, 50)

    train_dataset = TensorDataset(train_tokens_tensor, train_masks_tensor)
    train_sampler = RandomSampler(train_dataset)
    return DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

f=open('../../Downloads/MOOCCube/MOOCCube/relations/prerequisite-dependency.json', encoding='utf-8')
c_cnt, tot_cnt=0, 0
j_occ = eval(open('../../Downloads/PEBG-master/ednet/j_occ.txt', encoding='utf-8').read())
c_occ = eval(open('../../Downloads/PEBG-master/ednet/c_occ.txt', encoding='utf-8').read())
train_text = ["" for i in np.arange(len(j_occ))]

num_concepts, num_videos = len(j_occ.keys()), len(c_occ.keys())
print("Total # of concepts ", num_concepts)
print("Total # of videos ", num_videos)
adj = np.zeros((num_concepts,num_concepts))
f= open('../../Downloads/MOOCCube/MOOCCube/relations/prerequisite-dependency.json', encoding='utf-8')

for i in f:
    j=i.split()
    if j[0] in j_occ.keys() and j[1] in j_occ.keys():
        adj[j_occ[j[0]]][j_occ[j[1]]] = 1
f = open('../../Downloads/MOOCCube/MOOCCube/entities/concept.json', encoding='utf-8')
cnt=0
text_batch=["" for i in np.arange(len(j_occ))]
for i in f:
    j = (json.loads(i))
    st = j['name']
    if j['id'] not in j_occ.keys():
        continue
    st=st+'\n'+j['explanation']

    text_batch[j_occ[j['id']]] = (st)
# print(cnt)

f = open('../../Downloads/MOOCCube/MOOCCube/entities/video.json', encoding='utf-8')
cnt=0
text_batch_video=["" for i in np.arange(len(c_occ))]
for i in f:
    j = (json.loads(i))
    st = j['name']
    if j['id'] not in c_occ.keys():
        continue

    if 'text' in j.keys():
    #     if '又称' in j['explanation']:
    #         cnt+=1
        st=st+'\n'+'\n'.join(j['text'])

    text_batch_video[c_occ[j['id']]] = (st)


text_batch.extend(text_batch_video)

bert = BertModel.from_pretrained('bert-base-chinese').cuda()

cnt=0
bs=16
print("Length of Train text" ,len(text_batch))
train_dataloader = train_tensors(text_batch)
print(len(train_dataloader))
embs,x,tx = np.zeros(shape=(len(text_batch), 768)), np.zeros(shape=(400, 768)), np.zeros(shape=(25, 768))

# ***BERT embedings***#
# for step_num, batch_data in enumerate(train_dataloader):
#     if step_num%10000==0:
#         print(step_num)
#     token_ids, masks = tuple(t.to('cuda') for t in batch_data)
#     output, pooled_output = bert(token_ids.long(), attention_mask=masks, output_all_encoded_layers=False)
#     # embeddings = np.array(torch.mean(output,1).squeeze(1).detach().cpu())
#     embeddings = np.array(pooled_output.detach().cpu())
#     # print(embeddings.shape)
#
# #
#     embs[step_num*batch_size: step_num*batch_size+embeddings.shape[0],:] = embeddings

# ***Doc2Vec embedings***#
documents, embs=[],[]
# model.load('../../Downloads/179', 128)
for cnt, i in enumerate(text_batch):
    tokens = list(jieba.cut(i, cut_all=False))
    documents.append(tokens)

documents = []
sum1,sum2 =0,0
for cnt, i in enumerate(text_batch):
    tokens = list(jieba.cut(i, cut_all=False))
    if cnt<num_concepts:
        sum1+=len(tokens)
    else:
        sum2+=len(tokens)
    documents.append(tokens)
embs=[]
print("Token in Concepts" , sum1/num_concepts)
print("tokens in video", sum2/num_videos)
documents2 = [TaggedDocument(doc, [i]) for i, doc in enumerate(documents)]
model = Doc2Vec(documents2, vector_size=128, window=2, min_count=1, workers=4)
for document in text_batch:
    vector = model.infer_vector([doc for doc in document])
    embs.append(vector)
print(len(embs[0]))
embs  = np.array(embs)

# ***Word2vecembedings***#
# f = open('../../Downloads/zh/zh.tsv',encoding='utf-8')
# lines = f.readlines()
# w2v={}
# print(len(lines))
# i=0
# while i < len(lines):
#     # if i % 100==0:
#         # print(i)
#
#     j=lines[i].split('\t')
#     # print(j)
#
#     k=i+1
#     w=j[1]
#     vec =[]
#     for _it in j[2].split():
#         if _it[0] =='[':
#             if len(_it) == 1:
#                 continue
#             vec.append(float(_it[1:]))
#             continue
#         vec.append(float(_it))
#     while k<len(lines):
#
#         flag=0
#         for _it in lines[k].split():
#             if _it[-1] == ']':
#                 flag = 1
#                 if len(_it) == 1:
#                     break
#                 vec.append(float(_it[:-1]))
#
#                 break
#
#             vec.append(float(_it))
#
#         if flag==1:
#             break
#         k+=1
#
#     i=k+1
#
#     w2v[j[1]]=vec
# embs =[]
# print("embs length ", len(w2v['在']))
# for cnt,data in enumerate(text_batch):
#     if cnt%100==0:
#         print("No of lines processed", cnt)
#     tokens = jieba.cut(data,cut_all=False)
#     vectors = []
#     for token in tokens:
#         if token in w2v.keys():
#             vectors.append(w2v[token])
#     if len(vectors) == 0:
#         embs.append(np.zeros(300))
#     else:
#         embs.append(np.mean(vectors, axis=0))
#
# embs = np.array(embs)
# #


f=open('../../Downloads/MOOCCube/MOOCCube/entities/course.json', encoding='utf-8')
max_v_len , max_c_len=0,0
course_seq, pre_req = np.zeros((706,500)),[]
course2id = {}
position_ind, section_ind, mask = np.zeros((706,500)), np.zeros((706,500)), np.zeros((706,500))
pos_v,sec_v={},{}
for cnt2, i in enumerate(f):
    j=json.loads(i)
    if len(j['video_order'])> max_v_len:
        max_v_len = len(j['video_order'])
    pos, sec, cs  =[],[],[]
    pre_cnt, pre_k=0,-1
    for cnt, v in enumerate(j['video_order']):
        if v not in c_occ.keys():
            continue
        pos.append(cnt+1)
        pos_v[c_occ[v]]=cnt+1

        if int(j['chapter'][cnt].split('.')[0]) == pre_k:
            sec.append(pre_cnt+1)
            sec_v[c_occ[v]]=pre_cnt+1
            pre_cnt+=1

        else:
            pre_cnt=0
            sec.append(pre_cnt + 1)
            pre_k = int(j['chapter'][cnt].split('.')[0])
            sec_v[c_occ[v]] = pre_cnt + 1
        cs.append(c_occ[v])
    # if len(cs) ==0:
    #     continue
    if len(pos) ==0:
        continue
    position_ind[cnt2,-len(pos):]=(pos)
    section_ind[cnt2,-len(sec):] = (sec)

    course_seq[cnt2, -len(cs):] = (cs)
    mask[cnt2, -len(cs):] =1
    pre_k,len_k=-1,0
    for k in j['chapter']:
        if k.split('.')[0] == pre_k:
            len_k+=1
        else:
            if max_c_len<len_k:
                max_c_len=len_k
            len_k=1
            pre_k = k.split('.')[0]
    c2id[j['id']]=cnt2
f=open('course2id.txt','w')
f.write(str(c2id))
print(max_v_len)
print(max_c_len)
course_seq, position_ind, section_ind = np.array(course_seq), np.array(position_ind), np.array(section_ind)

model = course_model.model( num_concepts,torch.Tensor(embs[num_concepts:,:]),torch.Tensor(embs[:num_concepts,:]), max_v_len+1, max_c_len+1, 128).cuda()
adam_optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98))

pro_pro_coo = sparse.load_npz('../../Downloads/PEBG-master/ednet/pro_pro_sparse_ce.npz')
skill_skill_coo = sparse.load_npz('../../Downloads/PEBG-master/ednet/skill_skill_sparse_ce.npz')
pro_skill_coo = sparse.load_npz('../../Downloads/PEBG-master/ednet/pro_skill_sparse_ce.npz')
pro_pro_coo, skill_skill_coo, pro_skill_dense = pro_pro_coo.toarray(), skill_skill_coo.toarray(), pro_skill_coo.toarray()
(pro_pairs1, pro_pairs2) = np.nonzero(pro_pro_coo)
(skill_pairs1, skill_pairs2) = np.nonzero(skill_skill_coo)
arr_pro_pro, arr_pro_skill, arr_skill_skill, arr_course_course = [],[],[],[]

cp1,cp2,cp3=[],[],[]
for i in range(len(pro_pairs1)):
    cp1.append(pro_pairs1[i])
    cp2.append(pro_pairs2[i])
    cp3.append(int(1))
    n_neg=0
    while n_neg<3:
        k = random.randint(0,num_concepts-1)
        if pro_pro_coo[pro_pairs1[i]][k] == 0:
            cp1.append(pro_pairs1[i])
            cp2.append(k)
            cp3.append(int(0))
            n_neg+=1

cp1,cp2, cp3=np.array(cp1), np.array(cp2), np.array(cp3)
print("Length of CP1 for pro", cp1.shape)
cp1_,cp2_,cp3_=[],[],[]
for i in range(len(skill_pairs1)):
    cp1_.append(skill_pairs1[i])
    cp2_.append(skill_pairs2[i])
    cp3_.append(int(1))
    n_neg=0

    while n_neg<3:
        k = random.randint(0, num_videos-1)
        # print(k)
        if skill_skill_coo[skill_pairs1[i]][k] == 0:
            cp1_.append(skill_pairs1[i])
            cp2_.append(k)
            cp3_.append(int(0))
            n_neg+=1
cp1_,cp2_, cp3_=np.array(cp1_), np.array(cp2_), np.array(cp3_)
print("Length of CP1 for skill", cp1.shape)

pro_course_coo = sparse.load_npz('../../Downloads/PEBG-master/ednet/pro_course_sparse_ce.npz')
course_pairs_coo = pro_course_coo.toarray()
(course_pairs1, course_pairs2) = np.nonzero(course_pairs_coo)
cp13,cp23, cp33 = [],[],[]
for i in range(len(course_pairs1)):
    if np.sum(course_seq[course_pairs1[i]]) == 0 or np.sum(course_seq[course_pairs2[i]]) == 0:
        continue
    cp13.append(course_pairs1[i])
    cp23.append(course_pairs2[i])
    cp33.append(1)
    n_neg=0
    while n_neg<3:
        k = random.randint(0,705)
        if course_pairs_coo[course_pairs1[i]][k] == 0:
            cp13.append(course_pairs1[i])
            cp23.append(k)
            arr_course_course.append((course_pairs1[i], course_pairs2[i],k ))
            cp33.append(0)
            n_neg+=1

cp13,cp23, cp33=np.array(cp13), np.array(cp23), np.array(cp33)
print("Length of CP1", cp13.shape)


arr_pro_pro, arr_skill_skill, arr_pro_skill, arr_course_course = np.array(arr_pro_pro), np.array(arr_skill_skill), np.array(arr_pro_skill), np.array(arr_course_course)

print("CP1", len(cp1))
# Without concept
for i in range(0,len(cp1),2056):
    # if i%2056 == 0:

    c1, c2,c3 =torch.LongTensor(cp1[i:i+2056]), torch.LongTensor(cp2[i:i+2056]), torch.Tensor(cp3[i:i+2056])
    loss = model.forward_pro_pro(pro1=c1.cuda(), pro2=c2.cuda(), y=c3.cuda())
    adam_optimizer.zero_grad()
    loss.backward()
    adam_optimizer.step()
# Without video
print("CP2", len(cp1_))
for i in range(0,len(cp1_),2056):
    c1, c2, c3 = torch.LongTensor(cp1_[i:i + 2056]), torch.LongTensor(cp2_[i:i + 2056]), torch.Tensor(cp3_[i:i + 2056])
    loss = model.forward_skill_skill(c1.cuda(), c2.cuda(), c3.cuda())
    adam_optimizer.zero_grad()
    loss.backward()
    adam_optimizer.step()
# #
print("Max", np.max(section_ind))
print("CP3",len(cp13))
# for i in range(0,len(cp13),64):
#     c1=cp13[i:i+64]
#     c2 = cp23[i:i+64]
#     # print(course_seq[c1])
#     c1_seq = torch.LongTensor(course_seq[c1]).cuda()
#     c2_seq = torch.LongTensor(course_seq[c2]).cuda()
#     pos_seq1, pos_seq2 = torch.LongTensor(position_ind[c1]).cuda(), torch.LongTensor(position_ind[c2]).cuda()
#     sec_seq1, sec_seq2 = torch.LongTensor(section_ind[c1]).cuda(), torch.LongTensor(section_ind[c2]).cuda()
#     mask1, mask2 = torch.Tensor(mask[c1]).cuda(), torch.Tensor(mask[c2]).cuda()
#
#     loss=model.forward_course_course(c1_seq,c2_seq,pos_seq1, pos_seq2, sec_seq1, sec_seq2, mask1, mask2,torch.Tensor(cp33[i:i+64]).cuda())
#
#     adam_optimizer.zero_grad()
#     loss.backward()
#     adam_optimizer.step()

course_emb=np.zeros((706,128))
for i in range(0,706,32):

    c1_seq = torch.LongTensor(course_seq[i:i+32]).cuda()
    pos_seq1= torch.LongTensor(position_ind[i:i+32]).cuda()
    sec_seq1 = torch.LongTensor(section_ind[i:i+32]).cuda()

    c_emb=model.get_course_emb(c1_seq,pos_seq1, sec_seq1).detach().cpu()
    c_emb=np.array(c_emb)
    course_emb[i:i+c_emb.shape[0],:] = c_emb


f=open('../../anaconda3/envs/myenv/Lib/site-packages/course_embeddings.tsv','w')
for cnt,i in enumerate(course_emb):
    f.write(','.join([str(j) for j in i])+'\n')


f=open('../../anaconda3/envs/myenv/Lib/site-packages/concept_embeddings.tsv','w')
# concept_embs=embs[:num_concepts]
concept_embs = np.array(model.pro_embs.cpu().weight.data)
print(concept_embs.shape)
for cnt,i in enumerate(concept_embs):
    f.write(','.join([str(j) for j in i])+'\n')

video_embs = np.array(model.video_embs.cpu().weight.data)
pos_emb = np.array(model.pos_emb.cpu().weight.data)
sec_emb = np.array(model.sec_emb.cpu().weight.data)
f=open('../../anaconda3/envs/myenv/Lib/site-packages/entity_embeddings.tsv','w')
video_embs = embs[num_concepts:]
for cnt,i in enumerate(video_embs):

    f.write(','.join([str(j) for j in i])+',')
    f.write(','.join([str(j) for j in pos_emb[pos_v[cnt]]])+',')
    f.write(','.join([str(j) for j in sec_emb[sec_v[cnt]]])+'\n')
