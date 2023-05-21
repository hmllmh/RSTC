import torch
import math
import torch.nn.functional as F
from backend import get_backend
import numpy as np

def list_to_array(*lst):
    r""" Convert a list if in numpy format """
    if len(lst) > 1:
        return [np.array(a) if isinstance(a, list) else a for a in lst]
    else:
        return np.array(lst[0]) if isinstance(lst[0], list) else lst[0]

def sinkhorn_knopp(a, b, M, reg, numItermax=1000, stopThr=1e-9,
                   verbose=False, log=False, warn=True, u=None, v=None, h=None, reg2=1, log_alpha=10,  Hy='H1',
                   **kwargs):
    a, b, M = list_to_array(a, b, M)

    nx = get_backend(M, a, b)

    if len(a) == 0:
        a = nx.full((M.shape[0],), 1.0 / M.shape[0], type_as=M)
    if len(b) == 0:
        b = nx.full((M.shape[1],), 1.0 / M.shape[1], type_as=M)

    # init data
    dim_a = len(a)
    dim_b = b.shape[0]

    if len(b.shape) > 1:
        n_hists = b.shape[1]
    else:
        n_hists = 0

    if log:
        log = {'err': []}

    # we assume that no distances are null except those of the diagonal of
    # distances
    if u == None or v == None:
        if n_hists:
            u = nx.ones((dim_a, n_hists), type_as=M) / dim_a
            v = nx.ones((dim_b, n_hists), type_as=M) / dim_b
        else:
            u = nx.ones(dim_a, type_as=M) / dim_a
            v = nx.ones(dim_b, type_as=M) / dim_b

    K = nx.exp(M / (-reg))

    Kp = (1 / a).reshape(-1, 1) * K

    err = 1
    
    ######
    h.requires_grad_()
    ########========---------------
    for ii in range(numItermax):
        uprev = u
        vprev = v
        KtransposeU = nx.dot(K.T, u)
        v = b.detach() / KtransposeU
        u = 1. / nx.dot(Kp, v)        

        if (nx.any(KtransposeU == 0)
                or nx.any(nx.isnan(u)) or nx.any(nx.isnan(v))
                or nx.any(nx.isinf(u)) or nx.any(nx.isinf(v))):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Warning: numerical errors at iteration %d' % ii)
            u = uprev
            v = vprev
            break
        if ii % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            if n_hists:
                tmp2 = nx.einsum('ik,ij,jk->jk', u, K, v)
            else:
                # compute right marginal tmp2= (diag(u)Kdiag(v))^T1
                tmp2 = nx.einsum('i,ij,j->j', u, K, v)
            err = nx.norm(tmp2 - b.detach())  # violation of marginal
            if log:
                log['err'].append(err)

            if err < stopThr:
                break
            if verbose:
                if ii % 200 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(ii, err))

        #if step >= 0:  # 每一步都更新 0.819 2步更新一次 0.818 b的更新与p的正确与否有关，所以是不是等一些step以后再开始更新b 0.7454(有对比学习+swap predict)
        if Hy == 'H1':
            #if objective == 'saot' or objective == 'all':
            if True:
                tmpv = v.clone().detach()
                g = reg * torch.log(tmpv)
                bfh = 1000
                for i in range(10):  #
                    g[g==h] += 1e-5  # 处理b是nan的问题
                    delta = ((g - h)*log_alpha) ** 2 + 4 * (reg2 ** 2)  # cluster_num * 1
                    sqrt_delta = torch.sqrt(delta)
                    b = (((g - h)*log_alpha + 2*reg2) - sqrt_delta) / (2 * (g - h) * log_alpha)

                    fh = torch.sum(b) - 1
                    fh.backward()
                    h.data.sub_(fh.data / h.grad.data)  # 1 0.819 0.5 0.816 2 0.818
                    fh = torch.abs(fh)
                    if fh < bfh:
                        bfh = fh
                    else:
                        break
        elif Hy == 'H3':
            # update b H_3
            g = -reg * torch.log(v)
            b = torch.mul(b, torch.exp(g/reg2)) / torch.matmul(b, torch.exp(g/reg2))
        else:
            pass
    else:
        if warn:
            print("Sinkhorn did not converge. You might want to "
                          "increase the number of iterations `numItermax` "
                          "or the regularization parameter `reg`.")

    if log:
        log['niter'] = ii
        log['u'] = u
        log['v'] = v
        log['b'] = b.detach()

    if n_hists:  # return only loss
        res = nx.einsum('ik,ij,jk,ij->k', u, K, v, M)
        if log:
            return res, log
        else:
            return res

    else:  # return OT matrix

        if log:
            return u.reshape((-1, 1)) * K * v.reshape((1, -1)), log
        else:
            return u.reshape((-1, 1)) * K * v.reshape((1, -1))