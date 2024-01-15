import torch
import random

def mulhop_edge_sample(src_t, dst_t, N, alpha):
    edge_mh = torch.tensor([[-1, -1]]).t()
    i = 0
    j = 0

    sorted_src, indices_src = torch.sort(src_t[1])
    src_t = src_t[:, indices_src]

    sorted_dst, indices_dst = torch.sort(dst_t[0])
    dst_t = dst_t[:, indices_dst]

    print("src_t:", src_t)
    print("dst_t:", dst_t)
    print("src_t init:", src_t[1][0])
    print("dst_t init:", dst_t[0][0])

    while ((i <= N - 1) & (j <= N - 1)):
        cur_val_i = src_t[1][i]
        cur_val_j = dst_t[0][j]
        while (src_t[1][i] < cur_val_j):
            i += 1
            if (i == N):
                break
        cur_val_i = src_t[1][i]
        while (cur_val_i > dst_t[0][j]):
            j += 1
            if (j == N):
                break
        cur_val_j = dst_t[0][j]

        if ((i == N) | (j == N)):
            break

        if (cur_val_i != cur_val_j):
            continue
        same_val = cur_val_i
        slt_src = []
        slt_dst = []
        while (src_t[1][i] == same_val):
            slt_src.append(src_t[0][i])
            i += 1
            if (i == N):
                break
        while (dst_t[0][j] == same_val):
            slt_dst.append(dst_t[1][j])
            j += 1
            if (j == N):
                break
        slt_src = random.sample(slt_src, int(len(slt_src) * alpha))
        slt_dst = random.sample(slt_dst, int(len(slt_dst) * alpha))
        print("src:", slt_src)
        print("dst:", slt_dst)
        for src in slt_src:
            for dst in slt_dst:
                tmp_tensor = torch.tensor([[src, dst]]).t()
                edge_mh = torch.cat((edge_mh, tmp_tensor), dim=1)
        for i in range(0, N):
            tmp_tensor = torch.tensor([[i, i]]).t()
            edge_mh = torch.cat((edge_mh, tmp_tensor), dim=1)
    edge_mh = edge_mh[:, 1:]

    return edge_mh

def preprocess_khop(edge, k, N, alpha):
    res=[]
    cur_hop=edge
    for i in range(0,k):
        cur_hop=mulhop_edge_sample(cur_hop, edge, N, alpha)
        res.append(cur_hop)
    return res