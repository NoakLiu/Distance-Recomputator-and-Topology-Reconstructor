import random
import time
import torch

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
    
    # start=time.time()
    """
    for i in range(0,len(adj_lists[center_node])):
        if(adj_lists[int(center_node)][i]!=0):
            nghs.append(i)
    """
    #使用non-zero之后加快极其多
    print(torch.nonzero(adj_lists[center_node],as_tuple=True))
    nghs=torch.nonzero(adj_lists[center_node],as_tuple=True)[0].tolist()
    print(nghs)
    # end=time.time()
    # print("select for those non-zero elements:",end-start)
    
    nghs=_set(nghs) #{6,3,0,9,11}
    # print(nghs)
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
        print("h",h)
        per_hop_nghs=[]
        print(len(center_nodes))
        for i in range(0,len(center_nodes)):
            print("i",i)
            cnls=list(center_nodes[i]) #{这里是下一级的集合}

            print(cnls)
            start=time.time()
            ##从下一级中选择几个点进行下一步采样/从下一级中每个点分配随机数额进行采样
            ln=len(cnls)
            rand_ls=[random.randint(1,100) for j in range(0,ln)]
            rand_ls_sum=sum(rand_ls)
            p_ls=list(map(lambda i:i/rand_ls_sum,rand_ls))
            sample_num_ls=[elem*num_sample for elem in p_ls]
            end=time.time()
            print("the duration of selection is:",end-start)

            nx_level_elem=set()

            start=time.time()
            for j in range(0,ln):
                nx_level_elem=nx_level_elem.union(single_node_sample_around(cnls[j],int(sample_num_ls[j]),adj_lists))
                #print(single_node_sample_around(cnls[j],int(sample_num_ls[j]),adj_lists))
            #print(nx_level_elem)
            end=time.time()
            print("the duration of per node sample around is:",end-start)
            per_hop_nghs.append(list(nx_level_elem))
        center_nodes=per_hop_nghs
        if(h==k-1):
            return per_hop_nghs