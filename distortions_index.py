import numpy as np
from scipy.spatial.distance import cdist

from ga_k_means import Ga_K_Means

def f_k(dataset, k_max):
    Nd = 2
    A = [0]
    F = [0]
    S = [0]
    limit = 0.85

    for k in range(1, k_max + 1):
        g = Ga_K_Means()
        g.fit(dataset=dataset, k=k)

        M = np.array(g.best_fit[-1].Genes)  # cluster centers
        d = cdist(dataset, M)  # distance of all data to all cluster centers
        data_cluster_indices = np.argmin(d, axis=1)  # all data clusters , center indices
        # calculate all cluster members
        # ci_data_member = []
        Ii = []
        for i in range(k):
            # ci_data_member = np.array([dataset[j] for j in range(len(dataset)) if data_cluster_indices[j] == i])
            dci = np.array([d[j, i] for j in range(len(dataset)) if data_cluster_indices[j] == i])
            # a = np.multiply(dci, dci)
            # Ii.append(np.sum(a))
            Ii.append(np.dot(dci, dci))

        S.append(sum(Ii))
        # print(S[-1])

        if k == 1:
            A.append(0)

        elif k == 2:
            A.append(1 - (3 / (4 * Nd)))

        else:  # k > 2
            A.append(A[k - 1] + ((1 - A[k - 1]) / 6))

        if S[k - 1] == 0:
            F.append(1)
        else:
            F.append(S[k] / (A[k] * S[k-1]))

    res = [i for i in range(1, len(F)) if F[i] <= limit]

    return res, F, S, A


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import random
    from DeD import DeD
    # from dataset import dataset

    dataset0 = [[random.random() * 1.2 + 1] for i in range(50)]
    dataset1 = [[random.random()* 0.8 + 2.5] for i in range(50)]
    dataset = dataset1 + dataset0

    k , DeD = DeD(dataset=dataset)
    print('K: ', k, 'Ded:', DeD)

    res, F, S, A = f_k(dataset=dataset, k_max=15)
    print(res)
    print(F)
    print(S)
    print(A)


    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.plot(range(len(F) - 1), F[1:], marker='.')
    ax2.plot(range(len(DeD)), DeD, marker='x', c = 'r')

    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    ax1.scatter([0] * len(dataset), dataset, s=20, marker='.', c='r')

    plt.show()