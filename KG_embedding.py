import torch
import numpy as np
from models import KG
import json
from torch.optim import Adam
from tqdm import tqdm
import random
from scipy import sparse

torch.manual_seed(0)
random.seed(0)

def train(model, arr):


    total_loss =0
    for i in tqdm(range(0,len(arr), 2056)):
        triplet= torch.LongTensor(arr[i:i+2056,:]).cuda()
        loss = model.forward_pro_pro(triplet)
        total_loss+=loss
        model.zero_grad()
        loss.backward()
        optimizer.step()

    return  total_loss,model

def course_meta_data( n_course, course2id):
    f = open('../../Downloads/MOOCCube/MOOCCube/entities/course.json', encoding='utf-8')
    max_v_len, max_c_len = 0, 0
    course_seq, pre_req = np.zeros((n_course, 500)), []
    position_ind, section_ind = np.zeros((n_course, 500)), np.zeros((n_course, 500))
    pos_v, sec_v = {}, {}
    for  cnt2,i in enumerate(f):
        j = json.loads(i)
        if len(j['video_order']) > max_v_len:
            max_v_len = len(j['video_order'])
        pos, sec, cs = [], [], []
        pre_cnt, pre_k = 0, -1
        for cnt, v in enumerate(j['video_order']):
            if v not in c_occ.keys():
                continue
            pos.append(cnt + 1)
            pos_v[c_occ[v]] = cnt + 1

            if int(j['chapter'][cnt].split('.')[0]) == pre_k:
                sec.append(pre_cnt + 1)
                sec_v[c_occ[v]] = pre_cnt + 1
                pre_cnt += 1

            else:
                pre_cnt = 0
                sec.append(pre_cnt + 1)
                pre_k = int(j['chapter'][cnt].split('.')[0])
                sec_v[c_occ[v]] = pre_cnt + 1
            cs.append(c_occ[v] )

        if len(pos) == 0:
            continue
        position_ind[course2id[j['id']], -len(pos):] = (pos)
        section_ind[course2id[j['id']], -len(sec):] = (sec)

        course_seq[course2id[j['id']], -len(cs):] = (cs)
        pre_k, len_k = -1, 0
        for k in j['chapter']:
            if k.split('.')[0] == pre_k:
                len_k += 1
            else:
                if max_c_len < len_k:
                    max_c_len = len_k
                len_k = 1
                pre_k = k.split('.')[0]

    return np.array(course_seq), np.array(position_ind), np.array(section_ind), max_v_len , max_c_len

if __name__ == '__main__':
    ep=0

    pro_pro_coo = sparse.load_npz('../../Downloads/PEBG-master/ednet/pro_pro_sparse_ce.npz')
    pro_skill_coo = sparse.load_npz('../../Downloads/PEBG-master/ednet/pro_skill_sparse_ce.npz')
    pro_course_coo = sparse.load_npz('../../Downloads/PEBG-master/ednet/pro_course_sparse_ce.npz')
    course_course_coo = sparse.load_npz('../../Downloads/PEBG-master/ednet/course_course_sparse_ce.npz')
    pro_course_coo = sparse.load_npz('../../Downloads/PEBG-master/ednet/pro_course_sparse_ce.npz')
    pro_skill_dense = pro_skill_coo.toarray()
    pro_course_dense = pro_course_coo.toarray()

    course_course_dense = course_course_coo.toarray()

    f2 = open('../../Downloads/PEBG-master/ednet/j_occ_ce.txt', encoding='utf-8')
    j_occ = eval(f2.read())
    f2.close()
    f2 = open('../../Downloads/PEBG-master/ednet/c_occ_ce.txt', encoding='utf-8')
    c_occ = eval(f2.read())
    f2.close()
    f2 = open('course2id.txt', encoding='utf-8')
    course2id = eval(f2.read())
    f2.close()
    num_concepts = len(j_occ.keys())
    print("Concepts",num_concepts)
    num_skills = len(c_occ.keys())
    print(num_skills)
    num_course = 706
    print(num_course)
    print("Pro_skill_shape", pro_skill_dense.shape)



    arr_pro_pro, arr_pro_skill, arr_skill_skill, arr_course_course, arr_pro_course = [],[],[],[],[]


    for i in range(num_concepts):
        for j in range(num_skills):
            if i == j:
                continue

            if pro_skill_dense[i, j] ==1 :
                n_neg = 0
                while n_neg < 10:
                    # for k in range(num_concepts):
                    k = random.randint(0, num_skills - 1)
                    if pro_skill_dense[i,k] ==1:
                        continue
                    n_neg += 1
                    arr_pro_skill.append([int(i), int(j), int(k)])


    for i in range(num_concepts):
        for j in range(num_course):
            if i == j:
                continue

            if pro_course_dense[i, j] == 1:
                n_neg = 0
                while n_neg < 10:
                    # for k in range(num_concepts):
                    k = random.randint(0, num_course - 1)
                    if pro_course_dense[i, k] == 1:
                        continue
                    n_neg += 1
                    arr_pro_course.append([int(i), int(j), int(k)])



    arr_pro_skill, arr_pro_course =  np.array(arr_pro_skill), np.array(arr_pro_course)
    course_seq, position_ind, section_ind, max_v_len, max_c_len = course_meta_data(num_course, course2id)
    model = KG(num_concepts, num_skills, num_course, max_v_len+1, max_c_len+1).cuda()


    optimizer = Adam(model.parameters(), lr=0.001)
    features = np.load('../../Downloads/PEBG-master/ednet/pro_feat_ce.npz')['pro_feat']

    features_final =np.zeros((num_concepts, 1))
    for cnt,i in enumerate(features):
        features_final[cnt] = i[0]
    features_final = np.array(features_final)
    ep=0

    # Concept and course similarity
    for i in range(0, arr_pro_course.shape[0], 2056):
        triplet_pro_course = torch.LongTensor(arr_pro_course[i:i + 2056, :]).cuda()
        loss = model.forward_pro_course(triplet_pro_course)

        model.zero_grad()
        loss.backward()
        optimizer.step()
    #
    # ep=0
    # # while ep<200:

    # Concept and lecture similarity
    for i in range(0, arr_pro_skill.shape[0], 2056):
        triplet_pro_skill = torch.LongTensor(arr_pro_skill[i:i+2056,:]).cuda()
        loss=model.forward_pro_skill(triplet_pro_skill)

        pro_indices = torch.LongTensor(arr_pro_skill[i:i+2056,0])
        pro_indices = torch.unique(pro_indices)
        #
        pro_features = torch.Tensor(features_final[pro_indices]).cuda()

        model.zero_grad()
        loss.backward()
        optimizer.step()
    # # ep+=1
    # # ep = 0

    # Concept complexity level
    while ep < 100:
        for i in range(0, num_concepts, 2056):
            pro_indices = torch.LongTensor(np.arange(i, min(i + 2056, num_concepts)))

            pro_features = torch.Tensor(features_final[pro_indices]).cuda()
            loss = model.predict_difficulty(pro_indices.cuda(), pro_features)

            model.zero_grad()
            loss.backward()
            optimizer.step()
        ep += 1

    embs = model.state_dict()['pro_emb.weight']
    embs2 = model.pro_linear(embs)
    torch.save(embs2, 'embedding.pt')
    embs_course = model.state_dict()['course_emb.weight']
    torch.save(embs_course, 'course_embedding.pt')
    video_embs = model.state_dict()['skill_emb.weight'].detach().cpu()
    torch.save(video_embs, 'video_embedding.pt')

    # model = course_model.model( num_concepts,video_embs[:,:768],embs[:207,:].detach().cpu(),511, 66, 128)
    # model.load_state_dict(torch.load('model.pt'))
    # model.eval()
    # course_emb = np.zeros((706, 128))
    # for i in range(0, 706, 32):
    #     # print(course_seq[c1])
    #     c1_seq = torch.LongTensor(course_seq[i:i + 32])
    #     pos_seq1 = torch.LongTensor(position_ind[i:i + 32])
    #     sec_seq1 = torch.LongTensor(section_ind[i:i + 32])
    #
    #     c_emb = model.get_course_emb(c1_seq, pos_seq1, sec_seq1).detach().cpu()
    #     c_emb = np.array(c_emb)
    #     course_emb[i:i + c_emb.shape[0], :] = c_emb
    #
    # f = open('../../anaconda3/envs/myenv/Lib/site-packages/course_embeddings.tsv', 'w')
    # for cnt, i in enumerate(course_emb):
    #     f.write(','.join([str(j) for j in i]) + '\n')

