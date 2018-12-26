
from numpy import *
import nmf_utils as nu
import pickle


# nmf_embedding: learn embedding representation of diseases and genes
# using semi-supervised non-negative matrix factorization
# Note:
#   1. ppi_sp_file stores the data of shortest path length file of
#      protein pairs. We didn't upload this file because of too big size.
#      It can be obtained by calculating shortest path lengths base on
#      the data of protein pairs(i.e., ppi_data.txt).


dis_NI = {}   # key: dis name, value: dis id
dis_IN = {}   # key: dis id, value: dis name
gene_NI = {}  # key: gene name, value: gene id
gene_IN = {}  # key: gene id, value: gene name
eps = 1e-7
loss_list = []   # loss list

def init_para():
    global dis_NI, dis_IN, gene_NI, gene_IN, loss_list
    dis_NI = {}
    dis_IN = {}
    gene_NI = {}
    gene_IN = {}
    loss_list = []


def load_data(mat_file):
    V = []
    with open(mat_file, 'r') as fr:
        for line in fr:
            arr = line.strip().split("\t")
            V.append([float(x) for x in arr])
    return mat(V)

# load train and test edges and nodes from edge_train_test_file
def load_train_test_data(edge_train_test_file):
    print('load_train_test_data...')
    train_dic = {}  # key: 'dis', 'gene', 'edge', value: the set
    test_dic = {}  # key: 'dis', 'gene', 'edge', value: the set
    train_edge_set = set()
    test_edge_set = set()
    train_dis_set = set()
    train_gene_set = set()
    test_dis_set = set()
    test_gene_set = set()
    train_dis_dic = {}  # key: test dis_name, value: all the test genes of the dis
    test_dis_dic = {}  # key: train dis_name, value: all the train genes of the dis
    with open(edge_train_test_file, 'r') as fr:
        for line in fr:
            dis, gene, type = line.strip().split('\t')
            if type == 'train':
                if dis not in train_dis_dic:
                    a = set()
                    a.add(gene)
                    train_dis_dic[dis] = a
                else:
                    train_dis_dic[dis].add(gene)
                train_edge_set.add(dis+'\t'+gene)
                train_dis_set.add(dis)
                train_gene_set.add(gene)
            elif type == 'test':
                if dis not in test_dis_dic:
                    a = set()
                    a.add(gene)
                    test_dis_dic[dis] = a
                else:
                    test_dis_dic[dis].add(gene)
                test_edge_set.add(dis + '\t' + gene)
                test_dis_set.add(dis)
                test_gene_set.add(gene)
    train_dic['dis'] = train_dis_set
    train_dic['gene'] = train_gene_set
    train_dic['edge'] = train_edge_set
    train_dic['dis_dic'] = train_dis_dic
    test_dic['dis'] = test_dis_set
    test_dic['gene'] = test_gene_set
    test_dic['edge'] = test_edge_set
    test_dic['dis_dic'] = test_dis_dic
    return train_dic, test_dic


def cal_err(E):
    err = 0
    m, n = shape(E)
    for i in range(m):
        for j in range(n):
            err += E[i, j] * E[i, j]
    return err

# return a diagonal matrix whose entries are row summation of M
def cal_diag(M):
    M_diag = np.zeros([len(M), len(M)])
    for i in range(len(M)):
        M_diag[i, i] = sum(M[i])
    return M_diag

# V is user-item matrix
# B is item-item associations matrix
def train(V, B, k, iter, e, lam):
    global loss_list
    print('start train...')
    W, H = nu.init_WH(V, k)
    B_diag = cal_diag(B)
    V_current = V.copy()
    V_pre = np.dot(W, H.T)
    loss_list = []

    for x in range(iter):
        E = V_current - V_pre  # loss matrix
        V_pre = V_current.copy()
        err = np.linalg.norm(E)
        loss_list.append(err)
        if x % 10 == 0: print('iter:', x, 'err:', err)
        if err < e: break

        # update W
        # a = V * H
        a = np.dot(V, H)
        # b = W * H.T * H
        b = np.dot(np.dot(W, H.T), H)
        b[b == 0] = eps
        W = np.multiply(W, np.multiply(a, 1 / b))
        W[W < eps] = eps

        # update H
        # c = V.T * W + lam * B * H
        c = np.dot(V.T, W) + lam * np.dot(B, H)
        # d = H * W.T * W + lam * B_diag * H
        d = np.dot(np.dot(H, W.T), W) + lam * np.dot(B_diag, H)
        d[d == 0] = eps
        H = np.multiply(H, np.multiply(c, 1 / d))
        H[H < eps] = eps
        V_current = np.dot(W, H.T)

    return W, H

def row_norm(W):
    m, n = np.shape(W)
    minvals = np.min(W, axis=1).reshape(m, 1)
    maxvals = np.max(W, axis=1).reshape(m, 1)
    ranges = maxvals-minvals

    W_norm = W - np.tile(minvals, (1, n))
    ranges[ranges == 0] = eps
    W_norm = np.multiply(W_norm, 1 / (np.tile(ranges, (1, n))))
    return W_norm


# V is user-item matrix
# B is item-item associations matrix
def train_n(V, B, k, iter, e, lam):
    print('start train...')
    W, H = nu.init_WH(V, k)

    B_diag = cal_diag(B)
    V_current = V.copy()
    V_pre = np.dot(W, H.T)

    for x in range(iter):
        E = V_current - V_pre  # loss matrix
        V_pre = V_current.copy()
        err = np.linalg.norm(E)
        if x % 10 == 0: print('iter:', x, 'err:', err)
        if err < e: break
        # update W
        # a = V * H
        a = np.dot(V, H)
        # b = W * H.T * H
        b = np.dot(np.dot(W, H.T), H)
        b[b == 0] = eps
        W = np.multiply(W, np.multiply(a, 1 / b))
        W[W < eps] = eps
        W = row_norm(W)

        # update H
        # c = V.T * W + lam * B * H
        c = np.dot(V.T, W) + lam * np.dot(B, H)
        # d = H * W.T * W + lam * B_diag * H
        d = np.dot(np.dot(H, W.T), W) + lam * np.dot(B_diag, H)
        d[d == 0] = eps
        H = np.multiply(H, np.multiply(c, 1 / d))
        H[H < eps] = eps
        H = row_norm(H)
        V_current = np.dot(W, H.T)

    return W, H

def build_dic(train_dis_dic):
    global dis_NI, dis_IN, gene_NI, gene_IN
    d_counter, g_counter = 0, 0
    for dis, gset in train_dis_dic.items():
        if dis not in dis_NI:
            dis_NI[dis] = d_counter
            dis_IN[d_counter] = dis
            d_counter += 1
        for gene in gset:
            if gene not in gene_NI:
                gene_NI[gene] = g_counter
                gene_IN[g_counter] = gene
                g_counter += 1

def build_V(train_dis_dic):
    print('build V...')
    global dis_NI, dis_IN, gene_NI, gene_IN
    dis_num = len(dis_NI)
    gene_num = len(gene_NI)
    V = np.zeros([dis_num, gene_num])
    for dis, gset in train_dis_dic.items():
        did = dis_NI[dis]
        for gene in gset:
            gid = gene_NI[gene]
            V[did, gid] = 1
    return V

def build_gmat(ppi_sp_file, ppi_nodes_file, w):
    print('build gmat...')
    global gene_NI
    gene_num = len(gene_NI)
    gmat = np.zeros([gene_num, gene_num])
    ppi_node_IN = {}  # key: node id ,value: node name of ppi.
    # load ppi nodes dic
    with open(ppi_nodes_file, 'r') as fr:
        for line in fr:
            name, id = line.strip().split('\t')
            ppi_node_IN[id] = name
    with open(ppi_sp_file, 'r') as fr:
        for line in fr:
            gid1, gid2, sp = line.strip().split('\t')
            gname1 = ppi_node_IN[gid1]
            gname2 = ppi_node_IN[gid2]
            if gname1 not in gene_NI or gname2 not in gene_NI: continue
            g1 = gene_NI[gname1]
            g2 = gene_NI[gname2]
            weight = np.e ** (-float(sp) * float(sp))
            gmat[g1][g2] = weight * w
            gmat[g2][g1] = weight * w

    return gmat

def pre_out(test_dis_dic, train_dis_dic, train_gene_set, new_V, out_file):
    counter = 0
    test_dis_num = len(test_dis_dic)
    with open(out_file, 'w') as fw:
        fw.truncate()
        for test_d, d_genes in test_dis_dic.items():
            if counter % 100 == 0: print(counter, '/', test_dis_num, ':', test_d)
            counter += 1
            d_train_gene = train_dis_dic[test_d]
            # test_d is test dis, d_genes is the gene set of the test dis
            test_did = dis_NI[test_d]

            scores = []
            for i in range(len(train_gene_set)):
                score = new_V[test_did, i]
                gname = gene_IN[i]
                if score != 0: scores.append([score, gname])
            scores.sort(reverse=True)

            p_counter = 0
            d_test_gene_num = len(d_genes)
            for score, gene in scores:
                if gene not in d_train_gene:
                    p_counter += 1
                    if p_counter > max([d_test_gene_num * 2, 100]): break
                    fw.write(test_d + '\t' + gene + '\t' + str(score) + '\n')

def nmf(train_test_file, ppi_sp_file, ppi_nodes_file,
        out_file, param):
    global dis_NI, dis_IN, gene_NI, gene_IN
    init_para()
    e, iter, k, lam, w = param['e'], param['iter'],\
                         param['k'], param['lambda'], param['w']
    fw = open(out_file, 'w')
    fw.truncate()

    train_dic, test_dic = load_train_test_data(train_test_file)
    train_dis_dic = train_dic['dis_dic']
    test_dis_dic = test_dic['dis_dic']
    train_gene_set = train_dic['gene']
    build_dic(train_dis_dic)
    V = build_V(train_dis_dic)
    gmat = build_gmat(ppi_sp_file, ppi_nodes_file, w)
    W, H = train(V, gmat, k, iter, e, lam)
    new_V = np.dot(W, H.T)

    pre_out(test_dis_dic, train_dis_dic, train_gene_set, new_V, out_file)

def nmf_main():

    path = ''
    ppi_sp_file = path + 'blab_sp.txt'
    ppi_nodes_file = 'node_id.txt'
    train_test_file = 'cv10_of0.txt'
    out_file = 'out_top_cv0.txt'
    param = {'e': 1e-3, 'iter': 5000, 'k': 512, 'lambda': 1, 'w': 1}
    nmf(train_test_file, ppi_sp_file, ppi_nodes_file,
        out_file, param)


if __name__ == "__main__":
    pass
    nmf_main()
