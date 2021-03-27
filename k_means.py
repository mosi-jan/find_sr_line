import numpy as np
from scipy.spatial.distance import cdist


def k_means(dataset, centers=None, k=None):
    n = len(dataset)
    min_x = np.amin(dataset, axis=0)
    max_x = np.amax(dataset, axis=0)

    if centers is not None:
        k = len(centers)
        c = centers
    else:
        if k is not None:
            c = np.multiply(np.random.rand(k, len(dataset[0])), (max_x - min_x)) + min_x
        else:
            return None

    need_iter = True
    while need_iter:
        need_iter = False

        dxc = cdist(dataset, c)
        x_cluster_indices = np.argmin(dxc, axis=1)

        for i in range(k):
            ci_member = [dataset[j] for j in range(n) if x_cluster_indices[j] == i]
            # cM.append(np.mean(ci_member[i], axis=1))
            if len(ci_member) > 0:
                nc = np.mean(ci_member, axis=0)
                if c[i] != nc:
                    need_iter = True
                    c[i] = nc
            else:
                c[i] = np.random.rand(1, len(dataset[0])) * (max_x - min_x) + min_x
    return c


if __name__ == '__main__':
    dataset = [[1], [1.2], [1.5], [1.8], [2], [3.1], [3.5], [3.9], [5], [5.2], [5.4],
               [1.021], [1.58], [1.51], [10], [10.8], [8.6], [7.7], [1], [1.32], [1.7],
               [8.8], [9], [3.1], [5.5], [3.9], [5.4], [5.22], [5.4], [1.021], [1.58],
               [1.51], [1.8], [8.1], [7.5]]  # , [16.0]

    cf = [[1], [2]]
    # c = None
    centers = k_means(dataset=dataset, centers=cf, k=2)
    print(centers)
