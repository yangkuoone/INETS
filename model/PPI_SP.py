# -*- coding: utf-8 -*-

import numpy as np
from itertools import islice
import pickle as pk

geneDicNI = {}
geneDicIN = {}
spMat = None

# PPI_SP.py:
#    calculate the topological relevance of diseases and genes.
# Note:
#   varb_file is python data file that stores the information of shortest
#   path lengths of protein pairs. It can be generated based on protein-protein
#   interactions (i.e., ppi_data.txt).


def get_oneset_sp_statics(gene_set):
    global geneDicNI, geneDicIN, spMat
    inter_list = []

    for g in gene_set:
        if g not in inter_list and g in geneDicNI:
            inter_list.append(g)

    inter_num = len(inter_list)
    edgeNum = 0
    spSum = .0
    spSet = set()
    for i in range(inter_num):
        g1 = geneDicNI[inter_list[i]]
        for j in range(i + 1, inter_num-1):
            g2 = geneDicNI[inter_list[j]]
            sp = max([spMat[g1, g2], spMat[g2, g1]])
            if sp != 0:
                spSum = spSum + sp
                edgeNum = edgeNum + 1
                spSet.add(sp)

    if edgeNum != 0:
        avgSP = spSum / edgeNum
        maxSP = max(spSet)
        minSP = min(spSet)
        return edgeNum, avgSP, maxSP, minSP
    return 0, 0, 0, 0


def load_varb_file(varbFile):
    global geneDicNI, geneDicIN, spMat
    with open(varbFile, 'rb') as fr:
        geneDicNI = dict(pk.load(fr))
        geneDicIN = dict(pk.load(fr))
        spMat = np.matrix(pk.load(fr))
    return spMat



# get dis-gene relation
def get_dis_gene_rel(dis_gene_file):
    disGs = {}
    disNI = {}
    idDis = {}
    count = 0
    with open(dis_gene_file, 'r') as fr:
        for line in islice(fr, 0, None):
            dis, gene = line.strip().split('\t')
            disGs.setdefault(dis, set())
            disGs[dis].add(gene)
            if dis not in disNI:
                disNI[dis] = count
                idDis[count] = dis
                count = count + 1
    return disGs, idDis


def convt_weight(sp_list):
    weight_list = []
    for _, _, sp in sp_list:
        if sp == 0: continue
        weight_list.append(np.e ** (-sp*sp))
    return sum(weight_list)

# calculate SP info of two set
def get_SPofTwoSet(g1Set, g2Set):
    global geneDicNI, geneDicIN, spMat
    sp_dic = {}
    sp_list = []
    sp_statics = {}
    spSum = .0
    spSet = set()
    for g1Name in g1Set:
        if g1Name not in geneDicNI: continue
        g1Id = geneDicNI[g1Name]
        for g2Name in g2Set:
            if g2Name not in geneDicNI: continue
            g2Id = geneDicNI[g2Name]
            g1Id = int(g1Id)
            g2Id = int(g2Id)
            sp1 = spMat[g1Id, g2Id]
            sp2 = spMat[g2Id, g1Id]
            if g1Id == g2Id:
                sp = 0
                sp_list.append([g1Name, g2Name, sp])
                if sp in sp_statics:sp_statics[sp] += 1
                else:sp_statics[sp] = 1
                spSet.add(sp)
            if sp1 != 0 or sp2 != 0:
                if sp1 > sp2:
                    sp = sp1
                else:
                    sp = sp2
                sp_list.append([g1Name, g2Name, sp])
                if sp in sp_statics:sp_statics[sp] += 1
                else:sp_statics[sp] = 1
                spSum = spSum + sp
                spSet.add(sp)
    edge_num = len(sp_list)
    if edge_num != 0:
        sp_dic['min'] = min(spSet)
        sp_dic['max'] = max(spSet)
        sp_dic['avg'] = spSum / edge_num
        sp_dic['edge_num'] = edge_num
        sp_dic['sp_list'] = sp_list
        sp_dic['sp_statics'] = sp_statics
        return sp_dic
    else:
        sp_dic['edge_num'] = 0
        return sp_dic


def cal_sp(varb_file, dis_gene_file1, dis_gene_file2, thld,
                    out_file):
    print('getSpByTwoSet...')
    global geneDicNI, geneDicIN, spMat
    spMat = load_varb_file(varb_file)

    fw = open(out_file, 'w')
    fw.truncate()
    disGs1, idDis1 = get_dis_gene_rel(dis_gene_file1)
    disGs2, idDis2 = get_dis_gene_rel(dis_gene_file2)

    dis_num = len(disGs1)
    counter = 0
    for dis1, geneset1 in disGs1.items():
        print('counter', str(counter), '/', str(dis_num))
        counter += 1
        if len(geneset1) == 0: continue
        for dis2, geneset2 in disGs2.items():
            if len(geneset2) == 0: continue
            if dis1 == dis2: continue
            sp_dic = get_SPofTwoSet(geneset1, geneset2)
            if sp_dic['edge_num'] == 0: continue
            sp_list = sp_dic['sp_list']
            weight = round(convt_weight(sp_list), 3)
            edge_num = sp_dic['edge_num']
            min_sp = sp_dic['min']
            max_sp = sp_dic['max']
            avg_sp = round(sp_dic['avg'], 3)
            if weight < thld: continue
            fw.write('\t'.join([dis1, dis2, str(edge_num), str(min_sp), str(max_sp),
                                str(avg_sp), str(weight)])+'\n')
    fw.flush()
    fw.close()


def cal_sp_main():

    varb_file = 'varb_file.txt'
    dis_gene_file1 = 'cv10_of0_train.txt'
    dis_gene_file2 = 'all_gene.txt'
    out_file = 'cv0_genes_sp.txt'
    weight_thld = 0.01  # filter threshold
    cal_sp(varb_file, dis_gene_file1, dis_gene_file2, weight_thld, out_file)


if __name__ == '__main__':

    cal_sp_main()


