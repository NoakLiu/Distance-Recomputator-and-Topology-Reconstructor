import torch
from torch.autograd import Variable
from k_hop_nghs_sampling import sample_around_for_khop

def preprocess_khop(adj, hop_num,  num_sample):
    print("preprocess begin")
    nodes=[rowid for rowid in range(0,len(adj))]
    #row_indices = nodes #one node->one_node
    mask_list=[]
    for i in range(0,hop_num):
        print(i)
        print("here0")
        samp_neighs_for_khop=sample_around_for_khop(nodes,i+1, num_sample,adj)
        # column_indices=sample_around_for_khop(nodes,i+1, num_sample,adj)
        # print(samp_neighs_for_khop)
        print("here1")
        print(len(samp_neighs_for_khop))
        row_indices = [i for i in range(len(samp_neighs_for_khop)) for samp_neigh in samp_neighs_for_khop[i]]
        print("here2")
        column_indices = [samp_neigh for samp_neighs in samp_neighs_for_khop for samp_neigh in samp_neighs]
        mask = Variable(torch.zeros(adj.shape))
        print("col: ",len(column_indices))
        print("row: ",len(row_indices))
        mask[row_indices, column_indices] = 1
        mask_list.append(mask)
    print("preprocess end")
    return mask_list