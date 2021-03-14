import numpy as np
import pandas as pd
from time import sleep
import math
import random
from copy import deepcopy
from scipy.spatial import distance

from scipy.spatial.distance import cdist

import matplotlib.pyplot  as plt
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class Chromosome:
    def __init__(self, genes, fitness):
        self.Genes = genes
        self.Fitness = fitness

    def __gt__(self, other):
        return self.Fitness > other.Fitness


def CS_index(dataset, genes):
    centers = []
    for item in genes:
        if item[ - 1] == 1:
            centers.append(item[0:- 1])
    centers = np.array(centers)

    k = len(centers)
    n = len(dataset)

    if k < 2:
        return 10 * np.amax(cdist(dataset,dataset))
    # dist = np.zeros((len(dataset), len(centers)))
    # for x in dataset:
    #     for m in centers:
    #         dist[x,m] =np.linalg.norm(x-m)

    d = cdist(dataset,centers)
    x_center_indices  = np.argmin(d, axis=1)


    clusters = [[] for i in range(k)]
    for i in range(n):
        clusters[x_center_indices[i]].append(dataset[i])
    clusters = np.array(clusters)

    Dmax = np.zeros(k)
    for ci in range(k):
        if len(clusters[ci]) > 1:
            Dmax[ci] =np.mean(np.amax(cdist(clusters[ci], clusters[ci]), axis=1))
        else:
            Dmax[ci] = 10 * np.amax(cdist(dataset,dataset))

    c = cdist(centers, centers)
    for i in range(k):
        c[i,i] = np.inf

    Dmin = np.amin(c, axis=0)

    CS = np.sum(Dmax) / np.sum(Dmin)
    # CS = np.mean(Dmax) / np.mean(Dmin)
    # print(k, ': ', n, ': ',np.mean(Dmax),'\t', np.mean(Dmin), '\t', CS)
    # sleep(1)
    return CS


def fitness(dataset, genes):
    f = CS_index(dataset, genes)
    return f



class Ga:
    # ga params
    MaxIt = 200  # Maximum Number of Iterations
    nPop = 100  # Population Size
    pc = 0.8  # Crossover Percentage
    nc = 2 * round(pc * nPop / 2)  # Number of Offsprings (Parnets)
    pm = 0.3  # Mutation Percentage
    nm = round(pm * nPop)  # Number of Mutants
    gamma = 0.02
    mu = 0.05  # Mutation Rate
    beta = 8  # Selection Pressure

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

    def _create_Chromosime(self):
        # rnd = np.random
        # rnd.seed(seed=int(time.time()))
        threshold_set = [0, 1]

        genes = np.zeros((self.n_gene, self.dimension + 1))
        while True:
            genes[:, :self.dimension] = np.multiply(np.random.rand(self.n_gene, self.dimension),
                                                    (self.dimension_max_value - self.dimension_min_value)) \
                                        + self.dimension_min_value  # centroid
            genes[:, self.dimension] = np.random.choice(threshold_set, size=self.n_gene)  # clusters threshold
            # correct min 2 cluster
            while np.sum(genes[:, self.dimension]) < 2:
                genes[:, self.dimension] = np.random.choice(threshold_set, size=self.n_gene)  # clusters threshold

            fit = self.fitness(dataset=self.dataset, genes=genes)

            yield Chromosome(genes=genes, fitness=fit)

    def get_Chromosime(self):
        return self._create_Chromosime().__next__()

    def refresh_plot(self):
        self.ax.clear()
        if self.dimension == 1:
            self.ax.scatter([1 for i in range(len(self.C_data))], self.C_data, c='r', marker='v')
            # self.ax.scatter([1 for i in range(len(self.X_data))], self.X_data, c='b', marker='x')
            self.ax.scatter(self.X_data[:,1],self.X_data[:,0], c='b', marker='x')
            self.ax.plot([i/self.MaxIt - 1 for i in range(len(self.F_data))], self.F_data, color='b', marker='.')

        elif self.dimension == 2:
            self.ax.scatter(self.C_data[:,0],self.C_data[:,1], c='r', marker='v')
            self.ax.scatter(self.X_data[:, 1], self.X_data[:, 0], c='b', marker='x')
            self.ax.plot([i / self.MaxIt - 1 for i in range(len(self.F_data))], self.F_data, color='b', marker='.')

        elif self.dimension == 3:
            self.ax.scatter([self.C_data[:,0]],[self.C_data[:,1]],[self.C_data[:,2]], c='r', marker='v')
            self.ax.scatter([self.X_data[:,0]],[self.X_data[:,1]],[self.X_data[:,2]], c='b', marker='x')
            # a = [self.F_data[:,0]]
            # b = [self.F_data[:,1]]
            # c = [np.zeros(len(self.F_data))]
            # self.ax.plot(np.zeros(len(self.F_data)), self.F_data[:,1],self.F_data[:,0], color='b', marker='.')

            # self.ax.plot(np.zeros(len(self.F_data)), [i/self.MaxIt - 1 for i in range(len(self.F_data))], self.F_data, color='b', marker='.')


        plt.pause(0.001)


    def fit(self, dataset):
        self.dataset = np.array(dataset)
        try:
            self.dimension = len(self.dataset[0])
        except:
            self.dimension = 1

        self.n_gene =math.ceil(len(self.dataset)/2)
        self.dimension_min_value=np.amin(self.dataset,axis=0)
        self.dimension_max_value=np.amax(self.dataset,axis=0)

        if self.dimension == 1:
            self.ax = self.fig.add_subplot()
            for i in range(len(self.dataset)):
                self.X_data.append([self.dataset[i,0], 1])
        elif self.dimension == 2:
            self.ax = self.fig.add_subplot()
            self.X_data = dataset
        elif self.dimension == 3:
            self.ax = self.fig.add_subplot('111', projection='3d')
            self.X_data = dataset
        self.X_data = np.array(self.X_data)

        pop_list = np.array([self.get_Chromosime() for i in range(self.nPop)])

        # self.F_data = np.zeros((self.MaxIt,2))
        for i in range(self.MaxIt):
            self.best_fit.append(deepcopy(min( pop_list)))

            self.C_data = []
            for item in  self.best_fit[-1].Genes:
                if item[-1] == 1:
                    self.C_data.append(list(item[:-1]))
            self.C_data = np.array(self.C_data)
            # self.F_data.extend([self.best_fit[-1].Fitness * 10, (len(self.F_data) + 1) / self.MaxIt - 1])
            self.F_data.extend([self.best_fit[-1].Fitness * 10])
            # self.F_data[i]=[self.best_fit[-1].Fitness * 10, (len(self.F_data) + 1) *10/ self.MaxIt - 1]
            # self.F_data=np.array(self.F_data)
            self.refresh_plot()

            # create temp pop list
            temp_pop_list = np.zeros(self.nPop + self.nc + self.nm, type(Chromosome))  # + self.nm
            # temp_pop_list[0] = pop_list[0]
            temp_pop_list[0:self.nPop] = pop_list

            # for item in temp_pop_list:
            #     try:
            #         print(item.Genes)
            #     except:
            #         pass
            # print('-----------')
            # crossover
            temp_pop_list[self.nPop:self.nPop + self.nc] = self.crossover(pop_list=pop_list, nc=self.nc)
            temp_pop_list[self.nPop + self.nc:] = self.mutation(pop_list=pop_list, nm=self.nm)

            # for item in temp_pop_list:
            #     try:
            #         print(item.Genes)
            #     except:
            #         pass


            sort_list = np.sort(temp_pop_list)

            pop_list = deepcopy(sort_list[0:self.nPop])
            # print(len(pop_list), len(pop_list[0].Genes))
            print('pop_list:{}\t Genes count:{}\t best active gene count:{}\t best fitness:{}'.
                  format(len(pop_list), len(pop_list[0].Genes),len([i for i in self.best_fit[-1].Genes if i[-1]==1]), self.best_fit[-1].Fitness ))

    def crossover(self, pop_list, nc):
        child = np.zeros(nc, type(Chromosome))
        nc_pop = np.random.choice(pop_list, size=nc, replace=False)

        for i in range(int(nc/2)):
            p1 = deepcopy(nc_pop[2 * i])
            p2 = deepcopy(nc_pop[2 * i + 1])

            position = random.randint(1, self.n_gene)
            # print('position', position)

            l =deepcopy(p1.Genes[0:position])
            p1.Genes[0:position] = deepcopy(p2.Genes[0:position])
            p2.Genes[0:position] = l

            p1.Fitness = self.fitness(dataset=self.dataset, genes=p1.Genes)
            p2.Fitness = self.fitness(dataset=self.dataset, genes=p2.Genes)

            child[2 * i] = p1
            child[2 * i + 1] = p2

        return child

    def mutation(self, pop_list, nm):
        child = np.zeros(nm, type(Chromosome))
        nc_pop =deepcopy(np.random.choice(pop_list, size=nm))

        c = np.shape(nc_pop[0].Genes)
        coeff = np.random.rand(nm * self.dimension * self.n_gene) * 0.9
        coeff = np.resize(coeff, (nm, self.n_gene, self.dimension ))
        # coeff[:,-1] = 1

        for i in range(nm):
            a = np.shape(nc_pop[i].Genes[:,0])
            b = np.shape(coeff[i])
            nc_pop[i].Genes[:,:-1] *= coeff[i]
            nc_pop[i].Fitness = self.fitness(dataset=self.dataset, genes=nc_pop[i].Genes)
        # return np.array([self.get_Chromosime() for i in range(nm)])
        return nc_pop


if __name__ == '__main__':

    # dataset = [[random.randint(1,5), random.randint(15,25), random.randint(-10,-1)] for i in range(10)]
    # dataset = [[random.randint(1,5)] for i in range(5)]

    # dataset = [[1,1.2,1.5,1.4],[3.1,3.5,4.1],[5,5.2,7]]
    dataset = [[1],[1.2],[1.5],[1.8],[2], [3.1],[3.5],[3.9], [5],[5.2]]#,[5.4], [1.021],[1.58],[1.51],[10],[10.8],[8.8],[7.9],[0]]

    print(dataset)

    ga = Ga(fitness=fitness)
    ga.fit(dataset)


    print([item.Fitness for item in ga.best_fit])
    for i in range(len(ga.best_fit)):
        print(ga.best_fit[i].Genes)


    print(ga.best_fit[-1].Fitness)
    # ga.refresh_plot()
    # ga.fig.show()
    # sleep(10)
    plt.show()

