import random
import time
import torch
from torch.autograd import Variable

def single_node_sample_around(center_node, num_sample, adj_lists):
    """
    Input:
        center_node: node_id 5
        num_sample: 3
    Output:
        list of output nodes: {6,3,9}
    """

    # print(adj_lists.dtype)
    if(num_sample==0):
        return {}
    _set = set
    _sample = random.sample
    nghs=[]
    
    nghs=_set(nghs) #{6,3,0,9,11}
    if len(nghs)>=num_sample:
        samp_neighs=_set(_sample(nghs,num_sample,))
    else:
        samp_neighs=nghs
    #{6,3,9}
    return samp_neighs

def sample_around_for_khop(center_nodes, k, num_sample,adj_lists):
    """
    Return the kth layer neighborhood
    Input:
        center_nodes: a list of center nodes that to sample [0,1,..,2707]
        number_sample: the total number select from the next layer of the center nodes
        {5,2,3}->5:3,2:4,3:3
        {2,3}->2:5,3:5
    Output:
        [{6,3,9},{4,2,9,6},{2,1}]
    """
    center_nodes = [{center_node} for center_node in center_nodes]
    #print(center_nodes)
    for h in range(0,k):
        per_hop_nghs=[]
        for i in range(0,len(center_nodes)):
            cnls=list(center_nodes[i]) #{这里是下一级的集合}
            start=time.time()
            ##从下一级中选择几个点进行下一步采样/从下一级中每个点分配随机数额进行采样
            ln=len(cnls)
            rand_ls=[random.randint(1,100) for j in range(0,ln)]
            rand_ls_sum=sum(rand_ls)
            p_ls=list(map(lambda i:i/rand_ls_sum,rand_ls))
            sample_num_ls=[elem*num_sample for elem in p_ls]
            end=time.time()

            nx_level_elem=set()

            start=time.time()
            for j in range(0,ln):
                nx_level_elem=nx_level_elem.union(single_node_sample_around(cnls[j],int(sample_num_ls[j]),adj_lists))
            end=time.time()
            print("the duration of per node sample around is:",end-start)
            per_hop_nghs.append(list(nx_level_elem))
        center_nodes=per_hop_nghs
        if(h==k-1):
            return per_hop_nghs

def preprocess_khop(adj, hop_num,  num_sample):
    print("preprocess begin")
    nodes=[rowid for rowid in range(0,len(adj))]
    #row_indices = nodes #one node->one_node
    mask_list=[]
    start=time.time()
    for i in range(0,hop_num):
        print(i)
        samp_neighs_for_khop=sample_around_for_khop(nodes,i+1, num_sample,adj)
        # column_indices=sample_around_for_khop(nodes,i+1, num_sample,adj)
        # print(samp_neighs_for_khop)
        print(len(samp_neighs_for_khop))
        row_indices = [i for i in range(len(samp_neighs_for_khop)) for samp_neigh in samp_neighs_for_khop[i]]
        column_indices = [samp_neigh for samp_neighs in samp_neighs_for_khop for samp_neigh in samp_neighs]
        mask = Variable(torch.zeros(adj.shape))
        print("col: ",len(column_indices))
        print("row: ",len(row_indices))
        mask[row_indices, column_indices] = 1
        mask_list.append(mask)
    end=time.time()
    print("preprocess end")
    print("total preprocessing time is:{}s".format(end-start))
    return mask_list