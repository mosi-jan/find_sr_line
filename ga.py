import numpy as np
# import pandas as pd
from time import time
import math
import random
from copy import deepcopy
# from scipy.spatial import distance

from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class Chromosome:
    def __init__(self, genes, fitness):
        self.Genes = genes
        self.Fitness = fitness

    def __gt__(self, other):
        return self.Fitness > other.Fitness


def DB_index(dataset, genes):
    M = np.array([g[:-1] for g in genes if g[-1] >= 0.5])  # cluster centers
    k = len(M)  # cluster count
    n = len(dataset)  # data count

    penalty = 1 * np.amax(cdist(dataset, dataset))
    if k < 3:
        return penalty

    d = cdist(dataset, M)  # distance of all data to all cluster centers
    data_cluster_indices = np.argmin(d, axis=1)  # all data clusters , center indices

    ci_member = []
    Siq = []
    q = 2
    for i in range(k):
        ci_member.append([dataset[j] for j in range(n) if data_cluster_indices[j] == i])
        if len(ci_member[i]) >= 1:  # min cluster members
            a = [d[j, i] for j in range(n) if data_cluster_indices[j] == i]
            b = np.power(np.mean(np.power(a, q)), 1/q)
            Siq.append(b)
        else:
            Siq.append(penalty)

    t = 2.0
    Dijt = cdist(M, M, 'minkowski', t)

    Ritq = []
    for i in range(k):
        f = []
        for j in range(k):
            if j != i:
                if Dijt[i, j] == 0:
                    return penalty
                f.append((Siq[i] + Siq[j])/Dijt[i, j])
        Ritq.append(max(f))

    DB = np.mean(Ritq)

    # print('DB:{}\t ,cluster_members_count:{}'.format(DB, [len(item) for item in ci_member]))
    return DB


def CS_index(dataset, genes):
    M = np.array([g[:-1] for g in genes if g[-1] >= 0.5])  # cluster centers
    k = len(M)  # cluster count
    n = len(dataset)  # data count

    penalty = 1 * np.amax(cdist(dataset, dataset))
    if k < 3:
        return penalty

    d = cdist(dataset, M)  # distance of all data to all cluster centers
    data_cluster_indices = np.argmin(d, axis=1)  # all data clusters , center indices

    # calculate all cluster members
    ci_member = []
    Dmax = []
    for i in range(k):
        ci_member.append([dataset[j] for j in range(n) if data_cluster_indices[j] == i])
        Xi = np.array(ci_member[i])
        if len(Xi) >= 1:  # min cluster members
            max_xi = np.amax(cdist(Xi, Xi), axis=0)
            Dmax.append(np.mean(max_xi))
        else:
            Dmax.append(penalty)

    c = cdist(M, M)
    for i in range(k):
        c[i, i] = np.inf

    Dmin = np.amin(c, axis=0)
    # -----------
    if np.mean(Dmin) == 0:
        return penalty
    CS = np.mean(Dmax) / np.mean(Dmin)

    # print('Dmax:{} Dmin:{} cluster count:{} ci_member:{}'.format(Dmax, Dmin, k, ci_member))
    # print('CS:{}\t ,cluster_members_count:{}\t Dmax:{}\t Dmin:{}'
    # .format(CS, [len(item) for item in ci_member], Dmax, Dmin))
    return CS


def fitness(dataset, genes):
    CS_coeff = 0.55
    DB_coeff = 1 - CS_coeff

    f1 = DB_coeff * DB_index(dataset, genes)
    f2 = CS_coeff * CS_index(dataset, genes)
    f = f1 + f2
    # print('Fitness: {}\t DB:{}\t CS:{}'.format(f, f1,f2))
    return f


class Ga:
    # ga params
    MaxIt = 50  # Maximum Number of Iterations
    nPop = 250  # Population Size
    pc = 0.8  # Crossover Percentage
    nc = 2 * round(pc * nPop / 2)  # Number of Offsprings (Parents)
    pm = 0.6  # Mutation Percentage
    nm = round(pm * nPop)  # Number of Mutants
    gamma = 0.02
    mu = 0.05  # Mutation Rate
    beta = 8  # Selection Pressure
    pb = 0.4
    nb = round(pb * nPop)

    def __init__(self, fitness):
        self.fitness = fitness
        self.best_fit = []
        # calculate on fit function from dataset
        self.dataset = None
        self.dimension = None
        self.n_gene = None
        self.dimension_min_value = None
        self.dimension_max_value = None

        self.X_data = []
        self.C_data = []
        self.F_data = []
        # self.fig, self.ax = plt.subplots()
        self.fig = plt.figure()

    def _create_Chromosome(self):
        # rnd = np.random
        # rnd.seed(seed=int(time.time()))

        genes = np.random.rand(self.n_gene, self.dimension + 1)
        while True:
            genes[:, :self.dimension] = np.multiply(genes[:, :self.dimension],
                                                    (self.dimension_max_value - self.dimension_min_value)) \
                                        + self.dimension_min_value  # centroid

            fit = self.fitness(dataset=self.dataset, genes=genes)

            yield Chromosome(genes=genes, fitness=fit)

    def get_Chromosome(self):
        return self._create_Chromosome().__next__()

    def plot_update(self, dataset, best_fitness):
        self.ax.clear()
        # a = max(self.dimension_max_value)
        fitness = [max(self.dimension_max_value) * i.Fitness for i in best_fitness]

        M = np.array([g[:-1] for g in best_fitness[-1].Genes if g[-1] >= 0.5])  # cluster centers
        k = len(M)  # cluster count
        n = len(dataset)  # data count

        d = cdist(dataset, M)  # distance of all data to all cluster centers
        data_cluster_indices = np.argmin(d, axis=1)  # all data clusters , center indices

        # calculate all cluster members
        ci_member = []
        for i in range(k):
            ci_member.append([dataset[j] for j in range(n) if data_cluster_indices[j] == i])

        for i in range(k):
            self.ax.scatter([0]*len(ci_member[i]), ci_member[i], s=20, marker='.')
            self.ax.scatter([0], M[i], s=200, marker='_', c='r')

        # p_b =10
        # for p in range(p_b):
        #     if len(best_fitness) > p + 1:
        #         M1 = np.array([g[:-1] for g in best_fitness[-2-p].Genes if g[-1] >= 0.5])  # cluster centers
        #         for i in range(len(M1)):
        #             self.ax.scatter([0.1*(p+1)], M1[i], s=200, marker='_', c='k')

        self.ax.plot([i/self.MaxIt - 1.1 for i in range(len(fitness))], fitness, color='b', marker='.')

        plt.pause(0.0001)

    def fit(self, dataset):
        self.dataset = np.array(dataset)
        try:
            self.dimension = len(self.dataset[0])
        except:
            self.dimension = 1

        self.n_gene = math.ceil(len(self.dataset)/2)
        self.dimension_min_value = np.amin(self.dataset, axis=0)
        self.dimension_max_value = np.amax(self.dataset, axis=0)

        if self.dimension == 1:
            self.ax = self.fig.add_subplot()
        elif self.dimension == 2:
            self.ax = self.fig.add_subplot()
        elif self.dimension == 3:
            self.ax = self.fig.add_subplot('111', projection='3d')

        pop_list = np.array([self.get_Chromosome() for i in range(self.nPop)])

        # self.F_data = np.zeros((self.MaxIt,2))
        for i in range(self.MaxIt):
            # self.best_fit.append(deepcopy(min( pop_list)))
            # self.plot_update(dataset=dataset, best_fitness=self.best_fit)

            # self.C_data = []
            # for item in  self.best_fit[-1].Genes:
            #     if item[-1] >= 0.5:
            #         self.C_data.append(list(item[:-1]))
            # self.C_data = np.array(self.C_data)
            # # self.F_data.extend([self.best_fit[-1].Fitness * 10, (len(self.F_data) + 1) / self.MaxIt - 1])
            # self.F_data.extend([self.best_fit[-1].Fitness * 10])
            # # self.F_data[i]=[self.best_fit[-1].Fitness * 10, (len(self.F_data) + 1) *10/ self.MaxIt - 1]
            # # self.F_data=np.array(self.F_data)
            # self.refresh_plot()

            # create temp pop list
            temp_pop_list = np.zeros(self.nPop + self.nc + self.nm + self.nb, type(Chromosome))
            # temp_pop_list[0] = pop_list[0]
            temp_pop_list[0:self.nPop] = pop_list

            # crossover & mutation & new born
            temp_pop_list[self.nPop:self.nPop + self.nc] = self.crossover(pop_list=pop_list, nc=self.nc)
            temp_pop_list[self.nPop + self.nc:self.nPop + self.nc + self.nm] = self.mutation(pop_list=pop_list, nm=self.nm)
            temp_pop_list[self.nPop + self.nc + self.nm:] = np.array([self.get_Chromosome() for i in range(self.nb)])

            # select population
            pop_list = self.select(temp_pop_list, self.nPop)

            self.best_fit.append(deepcopy(min(pop_list)))
            self.plot_update(dataset=dataset, best_fitness=self.best_fit)

            print('pop_list:{}\t Genes count:{}\t best active gene count:{}\t best fitness:{}'.
                  format(len(pop_list), len(pop_list[0].Genes),
                         len([i for i in self.best_fit[-1].Genes if i[-1] >= 0.5]), self.best_fit[-1].Fitness))

            print(np.count_nonzero(pop_list[:] == min(pop_list)))
            # print(np.count_nonzero(pop_list[:] == self.best_fit[-1]))

    def crossover(self, pop_list, nc):
        child = np.zeros(nc, type(Chromosome))
        nc_pop = np.random.choice(pop_list, size=nc, replace=False)

        for i in range(int(nc/2)):
            p1 = deepcopy(nc_pop[2 * i])
            p2 = deepcopy(nc_pop[2 * i + 1])

            position = random.randint(1, self.n_gene)
            # print('position', position)

            l = deepcopy(p1.Genes[0:position])
            p1.Genes[0:position] = deepcopy(p2.Genes[0:position])
            p2.Genes[0:position] = l

            p1.Fitness = self.fitness(dataset=self.dataset, genes=p1.Genes)
            p2.Fitness = self.fitness(dataset=self.dataset, genes=p2.Genes)

            child[2 * i] = p1
            child[2 * i + 1] = p2

        return child

    def mutation(self, pop_list, nm):
        # child = np.zeros(nm, type(Chromosome))
        nc_pop = deepcopy(np.random.choice(pop_list, size=nm))

        coeff = np.random.rand(nm * self.n_gene, (self.dimension + 1)) + 0.5  # [0.5, 1)
        coeff = np.resize(coeff, (nm, self.n_gene, self.dimension + 1))

        min_limit = [self.dimension_min_value[:]]
        min_limit.append(0)
        max_limit = [self.dimension_max_value[:]]
        max_limit.append(0.9999999999999)

        for i in range(nm):
            nc_pop[i].Genes = nc_pop[i].Genes * coeff[i]
            for j in range(self.n_gene):
                for k in range(self.dimension + 1):
                    if nc_pop[i].Genes[j, k] > max_limit[k]:
                        nc_pop[i].Genes[j, k] = max_limit[k]
                    elif nc_pop[i].Genes[j, k] < min_limit[k]:
                        nc_pop[i].Genes[j, k] = min_limit[k]

            # nc_pop[i].Genes[:,:-1] = ((nc_pop[i].Genes[:, :-1] - self.dimension_min_value) *
            # coeff[i,:,:-1]) + self.dimension_min_value
            # nc_pop[i].Genes[:,-1] = ((nc_pop[i].Genes[:, -1] - self.dimension_min_value) *
            # coeff[i,:,-1]) + self.dimension_min_value
            nc_pop[i].Fitness = self.fitness(dataset=self.dataset, genes=nc_pop[i].Genes)
        # return np.array([self.get_Chromosome() for i in range(nm)])
        return nc_pop

    def select(self, pop_list, need_pop_number):
        n_pop_list = len(pop_list)
        a = np.zeros((n_pop_list, 2), type(float))
        a[:, 0] = np.random.choice(range(n_pop_list), size=n_pop_list, replace=False)
        a[:, 1] = np.array([pop_list[a[i, 0]].Fitness for i in range(n_pop_list)])

        max_fit_1 = np.amax(a[:, 1])
        # min_fit_1 = np.amin(a[:, 1])
        for i in range(n_pop_list):
            a[i, 1] = max_fit_1 / a[i, 1]

        # max_fit_2 = np.amax(a[:, 1])
        # min_fit_2 = np.amin(a[:, 1])
        # a[:,1] = a[:,1] / max_fit_1

        for i in range(n_pop_list - 1):
            a[i+1, 1] = a[i+1, 1]+a[i, 1]

        max_fit = np.amax(a[:, 1])

        k = 20
        res = np.zeros(need_pop_number, type(Chromosome))
        sort_list = np.sort(pop_list)

        res[0:k] = sort_list[0:k]
        # try:
        #     if 6*k > len(sort_list):
        #         res[k:2 * k] = np.random.choice(sort_list[k:], size=k, replace=False)
        #     else:
        #         res[k:2*k] = np.random.choice(sort_list[k:5*k], size=k, replace=False)
        # except:
        #     res[k:2*k] = np.random.choice(sort_list[k:], size=k, replace=False)

        p = round(need_pop_number * 0.02)
        if p < 1:
            p = 1
        print(p)
        i = 0
        while i < need_pop_number - k:  # 2*k:
            r = random.random() * (max_fit - 1) + 1
            for j in range(n_pop_list):
                if a[j, 1] >= r:
                    if np.count_nonzero(res[:i + k - 1] == pop_list[a[j, 0]]) < p:
                        res[i + k] = pop_list[a[j, 0]]
                        i += 1
                        # if np.count_nonzero(res[:i + 2*k-1]==pop_list[a[j, 0]]) > 0:
                        #     print(np.count_nonzero(res[:i + 2*k-1]==pop_list[a[j, 0]]))
                    break
                # if j == n_pop_list - 1:
                #     res[i] = pop_list[a[j, 0]]

        return res

    def refresh_plot(self):
        self.ax.clear()
        if self.dimension == 1:
            self.ax.scatter([1 for i in range(len(self.C_data))], self.C_data, c='r', s=100, marker='_')
            # self.ax.scatter([1 for i in range(len(self.X_data))], self.X_data, c='b', marker='x')
            self.ax.scatter(self.X_data[:, 1], self.X_data[:, 0], c='b', marker='.')
            self.ax.plot([i/self.MaxIt - 1 for i in range(len(self.F_data))], self.F_data, color='b', marker='.')

            # self.ax.scatter([[0, 0, 0],[-1, -1, -1]], [[1, 2, 5],[4, 5, 6]], s=100, marker='*', c=[1,2,3,4,5,6])

        elif self.dimension == 2:
            self.ax.scatter(self.C_data[:, 0], self.C_data[:, 1], c='r', marker='v')
            self.ax.scatter(self.X_data[:, 1], self.X_data[:, 0], c='b', marker='x')
            self.ax.plot([i / self.MaxIt - 1 for i in range(len(self.F_data))], self.F_data, color='b', marker='.')

        elif self.dimension == 3:
            self.ax.scatter([self.C_data[:, 0]], [self.C_data[:, 1]], [self.C_data[:, 2]], c='r', marker='v')
            self.ax.scatter([self.X_data[:, 0]], [self.X_data[:, 1]], [self.X_data[:, 2]], c='b', marker='x')
            # a = [self.F_data[:,0]]
            # b = [self.F_data[:,1]]
            # c = [np.zeros(len(self.F_data))]
            # self.ax.plot(np.zeros(len(self.F_data)), self.F_data[:,1],self.F_data[:,0], color='b', marker='.')
            # self.ax.plot(np.zeros(len(self.F_data)), [i/self.MaxIt - 1 for i
            # in range(len(self.F_data))], self.F_data, color='b', marker='.')
        plt.pause(0.001)


if __name__ == '__main__':
    start_time = time()

    # dataset = [[random.randint(1,5), random.randint(15,25), random.randint(-10,-1)] for i in range(10)]
    # dataset = [[random.randint(1,5)] for i in range(5)]

    # dataset = [[1,1.2,1.5,1.4],[3.1,3.5,4.1],[5,5.2,7]]
    # dataset = [[1],[1.2],[1.5],[1.8],[2], [3.1],[3.5],[3.9], [5],[5.2],[5.4],
    # [1.021],[1.58],[1.51],[10],[10.8],[8.6],[7.9]]
    dataset = [[1], [1.2], [1.5], [1.8], [2], [3.1], [3.5], [3.9], [5], [5.2], [5.4],
               [1.021], [1.58], [1.51], [10], [10.8], [8.6], [7.7], [1], [1.32], [1.7],
               [8.8], [9], [3.1], [5.5], [3.9], [5.4], [5.22], [5.4], [1.021], [1.58],
               [1.51], [16.0], [1.8], [8.1], [7.5]]

    print(dataset)

    ga = Ga(fitness=fitness)
    ga.fit(dataset)

    print([item.Fitness for item in ga.best_fit])
    # for i in range(len(ga.best_fit)):
    #     print(ga.best_fit[i].Genes)

    print(ga.best_fit[-1].Genes)
    print(ga.best_fit[-1].Fitness)

    M = np.array([g[:-1] for g in ga.best_fit[-1].Genes if g[-1] >= 0.5])  # cluster centers
    k = len(M)  # cluster count

    d = cdist(np.array(dataset), M)  # distance of all data to all cluster centers
    data_cluster_indices = np.argmin(d, axis=1)  # all data clusters , center indices
    # calculate all cluster members
    ci_data_member = []
    for i in range(k):
        ci_data_member.append([dataset[j] for j in range(len(dataset)) if data_cluster_indices[j] == i])
    print([len(item) for item in ci_data_member])
    # ga.refresh_plot()
    # ga.fig.show()
    # sleep(10)
    print('run time:{}'.format(time()-start_time))
    plt.show()
