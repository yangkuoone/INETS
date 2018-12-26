import numpy as np

# Fusion algorithm: integrate the information of embedding features
#  and shortest path length of PPI network.

class Fusion(object):
    def __init__(self):
        pass
        self.pre_file = ''    # embedding information
        self.sp_result_file = ''    # shortest path length information
        self.out_file = ''
        self.sp_weight = 0
        self.dis_score_raw = dict()
        self.dis_dict_sp = dict()
        self.sp_fusion_main()

    def sp_fusion_main(self):

        self.pre_file = 'out_top_cv0.txt'
        self.sp_result_file = 'cv0_genes_sp.txt'
        self.out_file = 'out_top_cv0.txt'
        self.sp_weight = 0.8
        self.sp_fusion()


    def sp_fusion(self):
        self.load_raw_result()
        self.load_sp_result()
        num = len(self.dis_score_raw)
        counter = 1
        with open(self.out_file, 'w') as fw:
            for dis, gene_dict in self.dis_score_raw.items():
                print('counter', counter, '/', num)
                counter += 1
                gene_score = []
                for gene, score in gene_dict.items():
                    score_sp = self.dis_dict_sp.get((dis, gene))
                    if score_sp is not None:
                        score += score_sp
                    gene_score.append([score, gene])
                gene_score.sort(reverse=True)
                for score, gene in gene_score:
                    fw.write('\t'.join([dis, gene, str(score)])+'\n')

    def load_sp_result(self):
        print('load sp result...')
        with open(self.sp_result_file, 'r') as fr:
            for line in fr:
                arr = line.strip().split('\t')
                dis = arr[0]
                gene = arr[1]
                temp = float(arr[-1])
                weight = np.e ** (-temp * temp)
                self.dis_dict_sp[(dis, gene)] = weight*self.sp_weight

    def load_raw_result(self):
        print('load raw result...')
        with open(self.pre_file, 'r') as fr:
            for line in fr:
                dis, gene, score = line.strip().split('\t')
                self.dis_score_raw.setdefault(dis, dict())
                self.dis_score_raw[dis][gene] = float(score)


if __name__ == '__main__':
    fu = Fusion()



