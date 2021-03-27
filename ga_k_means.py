import numpy as np
from scipy.spatial.distance import cdist

import random

import matplotlib.pyplot as plt
from copy import deepcopy

from k_means import k_means


def sum_min_distance_index(dataset, genes):
    M = np.array(genes)  # cluster centers
    d = cdist(dataset, M)  # distance of all data to all cluster centers

    return np.sum(np.amin(d, axis=1))


def default_fitness(dataset, genes):
    genes = k_means(dataset=dataset, centers=genes)

    f = sum_min_distance_index(dataset, genes)
    return f


class Chromosome:
    def __init__(self, genes, fitness):
        self.Genes = genes
        self.Fitness = fitness

    def __gt__(self, other):
        return self.Fitness > other.Fitness


class Ga_K_Means:
    # ga params
    MaxIt = 3  # Maximum Number of Iterations
    nPop = 200  # Population Size
    pc = 0.9  # Crossover Percentage
    nc = 2 * round(pc * nPop / 2)  # Number of Offsprings (Parents)
    pm = 0.3  # Mutation Percentage
    nm = round(pm * nPop)  # Number of Mutants
    gamma = 0.02
    mu = 0.05  # Mutation Rate
    beta = 8  # Selection Pressure
    pb = 0.3
    nb = round(pb * nPop)

    def __init__(self, fitness=None, chart=None):
        self.chart = chart
        if fitness is None:
            self.fitness = default_fitness
        else:
            self.fitness = fitness
        self.best_fit = []
        # calculate on fit function from dataset
        self.dataset = None
        self.dimension = None
        self.n_gene = None
        self.dimension_min_value = None
        self.dimension_max_value = None
        if self.chart is not None:
            self.fig = plt.figure()

    def _create_Chromosome(self):
        # rnd = np.random
        # rnd.seed(seed=int(time.time()))

        genes = np.random.rand(self.n_gene, self.dimension)
        while True:
            genes = np.multiply(genes, (self.dimension_max_value - self.dimension_min_value)) + self.dimension_min_value  # centroid

            fit = self.fitness(dataset=self.dataset, genes=genes)

            yield Chromosome(genes=genes, fitness=fit)

    def get_Chromosome(self):
        return self._create_Chromosome().__next__()

    def plot_update(self, dataset, best_fitness):
        self.ax.clear()
        # a = max(self.dimension_max_value)
        fitness = [ i.Fitness for i in best_fitness]

        M = np.array([g[:-1] for g in best_fitness[-1].Genes if g[-1] >= 0.5])  # cluster centers
        M = np.array(best_fitness[-1].Genes)  # cluster centers
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
        coeff = np.array(max(self.dimension_max_value) / max(fitness))
        self.ax.plot([i/self.MaxIt - 1.1 for i in range(len(fitness))], fitness * coeff, color='b', marker='.')

        plt.pause(0.0001)

    def fit(self, dataset, k):
        self.dataset = np.array(dataset)
        try:
            self.dimension = len(self.dataset[0])
        except:
            self.dimension = 1

        # self.n_gene = math.ceil(len(self.dataset)/2)
        self.n_gene = k
        self.dimension_min_value = np.amin(self.dataset, axis=0)
        self.dimension_max_value = np.amax(self.dataset, axis=0)

        if self.chart is not None:
            if self.dimension == 1:
                self.ax = self.fig.add_subplot()
            elif self.dimension == 2:
                self.ax = self.fig.add_subplot()
            elif self.dimension == 3:
                self.ax = self.fig.add_subplot('111', projection='3d')

        pop_list = np.array([self.get_Chromosome() for i in range(self.nPop)])

        for i in range(self.MaxIt):
            temp_pop_list = np.zeros(self.nPop + self.nc + self.nm + self.nb, type(Chromosome))
            temp_pop_list[0:self.nPop] = pop_list

            # crossover & mutation & new born
            temp_pop_list[self.nPop:self.nPop + self.nc] = self.crossover(pop_list=pop_list, nc=self.nc)
            temp_pop_list[self.nPop + self.nc:self.nPop + self.nc + self.nm] = self.mutation(pop_list=pop_list, nm=self.nm)
            temp_pop_list[self.nPop + self.nc + self.nm:] = np.array([self.get_Chromosome() for i in range(self.nb)])

            # select population
            pop_list = self.select(temp_pop_list, self.nPop)

            self.best_fit.append(deepcopy(min(pop_list)))
            if self.chart is not None:
                self.plot_update(dataset=self.dataset, best_fitness=self.best_fit)

            print('pop_list:{}\t Genes count:{}\t best active gene count:{}\t best fitness:{}'.
                  format(len(pop_list), len(pop_list[0].Genes),
                         len([i for i in self.best_fit[-1].Genes if i[-1] >= 0.5]), self.best_fit[-1].Fitness))


            # print('pop_list:{}\t Genes count:{}\t best active gene count:{}\t best fitness:{} CS_index:{} DB_index:{}'.
            #       format(len(pop_list), len(pop_list[0].Genes),
            #              len([i for i in self.best_fit[-1].Genes if i[-1] >= 0.5]), self.best_fit[-1].Fitness,
            #              CS_index(dataset=self.dataset, genes=self.best_fit[-1].Genes),
            #              DB_index(dataset=self.dataset, genes=self.best_fit[-1].Genes)))
            #
            # print(np.count_nonzero(pop_list[:] == min(pop_list)))
            # print(np.count_nonzero(pop_list[:] == self.best_fit[-1]))

        return (self.best_fit[-1].Genes, self.best_fit[-1].Fitness)

        # return [self.best_fit[-1].Fitness,
        #         CS_index(dataset=self.dataset, genes=self.best_fit[-1].Genes),
        #         DB_index(dataset=self.dataset, genes=self.best_fit[-1].Genes)]

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

        coeff = np.random.rand(nm * self.n_gene, self.dimension) + 0.5  # [0.5, 1)
        coeff = np.resize(coeff, (nm, self.n_gene, self.dimension))

        for i in range(nm):
            nc_pop[i].Genes = nc_pop[i].Genes * coeff[i]
            for j in range(self.n_gene):
                for k in range(self.dimension):
                    if nc_pop[i].Genes[j, k] > self.dimension_max_value[k]:
                        nc_pop[i].Genes[j, k] = self.dimension_max_value[k]
                    elif nc_pop[i].Genes[j, k] < self.dimension_min_value[k]:
                        nc_pop[i].Genes[j, k] = self.dimension_min_value[k]

            nc_pop[i].Fitness = self.fitness(dataset=self.dataset, genes=nc_pop[i].Genes)
        return nc_pop

    def select(self, pop_list, need_pop_number):
        n_pop_list = len(pop_list)
        a = np.zeros((n_pop_list, 2), type(float))
        a[:, 0] = np.random.choice(range(n_pop_list), size=n_pop_list, replace=False)
        a[:, 1] = np.array([pop_list[a[i, 0]].Fitness for i in range(n_pop_list)])

        max_fit_1 = np.amax(a[:, 1])
        # min_fit_1 = np.amin(a[:, 1])
        for i in range(n_pop_list):
            if a[i, 1] == 0:
                a[i, 1] = 0.0001
            a[i, 1] = max_fit_1 / a[i, 1]

        for i in range(n_pop_list - 1):
            a[i + 1, 1] = a[i + 1, 1] + a[i, 1]

        max_fit = np.amax(a[:, 1])

        k = 2
        res = np.zeros(need_pop_number, type(Chromosome))
        sort_list = np.sort(pop_list)

        res[0:k] = sort_list[0:k]

        p = round(need_pop_number * 0.05)
        if p < 1:
            p = 1
        # print(p)
        i = 0
        while i < need_pop_number - k:  # 2*k:
            r = random.random() * (max_fit - 1) + 1
            for j in range(n_pop_list):
                if a[j, 1] >= r:
                    if np.count_nonzero(res[:i + k - 1] == pop_list[a[j, 0]]) < p:
                        res[i + k] = pop_list[a[j, 0]]
                        i += 1
                    break
        return res


if __name__ == '__main__':
    import DeD
    from time import time

    start_time = time()

    # dataset = [[random.randint(1,5), random.randint(15,25), random.randint(-10,-1)] for i in range(10)]
    # dataset = [[random.randint(1,5)] for i in range(5)]

    # dataset = [[1,1.2,1.5,1.4],[3.1,3.5,4.1],[5,5.2,7]]
    # dataset = [[1],[1.2],[1.5],[1.8],[2], [3.1],[3.5],[3.9], [5],[5.2],[5.4],
    # [1.021],[1.58],[1.51],[10],[10.8],[8.6],[7.9]]

    # dataset = [[1], [1.2], [1.5], [1.8], [2], [3.1], [3.5], [3.9], [5], [5.2], [5.4],
    #            [1.021], [1.58], [1.51], [10], [10.8], [8.6], [7.7], [1], [1.32], [1.7],
    #            [8.8], [9], [3.1], [5.5], [3.9], [5.4], [5.22], [5.4], [1.021], [1.58],
    #            [1.51], [1.8], [8.1], [7.5], [16.0], [17.01]]#

    dataset = [[1], [1.2], [1.5], [1.8], [2], [3.1], [3.5], [3.9], [5], [5.2], [5.4],
               [1.021], [1.58], [1.51], [10], [10.8], [8.6], [7.7], [1], [1.32], [1.7],
               [8.8], [9], [3.1], [5.5], [3.9], [5.4], [5.22], [5.4], [1.021], [1.58],
               [1.51], [1.8], [8.1], [7.5]]#, [16.0]


    print(dataset)
    # dataset = [[1], [2], [3], [4], [3.5], [4.6]]

    a=set([d[0] for d in dataset])
    k = len(a)
    print('unique data number:', k)
    # k = 12

    k = DeD.DeD(dataset=dataset)[0] + 2
    print('k:', k - 2)

    step_ga_obj = []
    for i in range(k):
        # ga = Ga_K_Means()
        step_ga_obj.append(Ga_K_Means())
        step_ga_obj[-1].fit(dataset, i + 1)


    print([g.best_fit[-1].Fitness for g in step_ga_obj])

    print(step_ga_obj[-1].best_fit[-1].Genes)
    print(step_ga_obj[-1].best_fit[-1].Fitness)

    # M = np.array([g[:-1] for g in ga.best_fit[-1].Genes if g[-1] >= 0.5])  # cluster centers
    M = np.array(step_ga_obj[-1].best_fit[-1].Genes)  # cluster centers
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
    print(step_ga_obj)


    fit_data = [g.best_fit[-1].Fitness for g in step_ga_obj]

    # fig = plt.figure()
    fig2 = plt.figure()
    # fig3 = plt.figure()
    # ax = fig.add_subplot()
    ax2 = fig2.add_subplot()
    # ax3 = fig3.add_subplot()
    ax2.plot(range(k), fit_data, marker='.')
    # ax2.plot(range(k), best_fits[:, 0], marker='.')
    # ax3.plot(range(k), best_fits[:, 1], marker='.')
    #
    # ax.plot(range(k-1), best_fits[1:, 2], marker='.')
    # ax.plot(range(k-1), best_fits[1:, 3], marker='.')
    # ax.plot(range(k-1), best_fits[1:, 3], marker='.')

    plt.show()
