import numpy as np
from scipy import linalg
import math


def mahalanobis0(x=None, data=None, cov=None):
    """Compute the Mahalanobis Distance between each row of x and the data
    x    : vector or matrix of data with, say, p columns.
    data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
    cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
    """
    x_minus_mu = x - np.mean(data)
    if not cov:
        a = data.T
        cov = np.cov(a)
        # cov = np.cov(data.values.T)
    inv_covmat = linalg.inv(cov)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    return mahal.diagonal()


def mahalanobis(x, y, cov=None):
    x_mean = np.mean(x)
    x_minus_mn = x - x_mean

    Covariance = np.cov(np.transpose(y))
    inv_covmat = np.linalg.inv(Covariance)
    D_square = np.dot(np.dot(x_minus_mn, inv_covmat), np.transpose(x_minus_mn))
    return D_square


def DeD(dataset, max_k=20):
    dataset = np.sort(dataset, axis=0)
    n = len(dataset)

    if max_k > n:
        max_k = n

    # step 1
    # MD(x; Xi)=[1 + (x−Xi)]−1
    x_mean = np.mean(dataset, axis=0)
    x_minus_mn = dataset - x_mean
    abs_x_minus_mn = np.abs(x_minus_mn)
    Di = np.divide(1, abs_x_minus_mn + 1)
    DM = max(Di)

    # step 2
    DeD = []
    delta_k = []

    for k in range(2, max_k):
        # k = t + 2

        cl = math.ceil(n / k)
        if cl * k > n:
            cl -= 1
        # start = 0
        end = 0

        for i in range(k):
            start = end
            end = start + cl
            if i == k - 1:
                end = n

            DMk = max(Di[start:end])
            delta_k.append(np.mean(np.abs(dataset[start:end] - DMk)))

        DW = np.mean(delta_k)
        delta = np.mean(np.abs(Di - DM))
        DB = delta - DW
        DeD.append(DW - DB)

    K = np.argmax(DeD) + 2

    return K, DeD


def DeD0(dataset):
    k = 0
    # step 1
    # MD(x; Xi)=[1 + (x−Xi)TCov(X)−1(x−Xi)]−1
    x_mean = np.mean(dataset, axis=0)
    x_minus_mn = dataset - x_mean

    Covariance = np.cov(dataset)
    # inv_covmat = np.matrix_power(dataset, -1)
    # inv_covmat = linalg.inv(Covariance)
    inv_covmat = np.linalg.inv(Covariance)

    t = np.dot(np.transpose(x_minus_mn), inv_covmat)
    D_square = np.dot(t, x_minus_mn)

    a = 1 + D_square
    MD =np.linalg.inv(a)

    # step 2

    return MD


if __name__ == '__main__':
    # dataset = [[1], [1.2], [1.5], [1.8], [2], [3.1], [3.5], [3.9], [5], [5.2], [5.4],
    #                [1.021], [1.58], [1.51], [10], [10.8], [8.6], [7.7], [1], [1.32], [1.7],
    #                [8.8], [9], [3.1], [5.5], [3.9], [5.4], [5.22], [5.4], [1.021], [1.58],
    #                [1.51], [1.8], [8.1], [7.5]]#, [16.0]


    # dataset = [[1,2],[2,3],[3,3]]
    # dataset = [[1,0],[2,1]]
    # dataset = [[1],[2],[3],[4],[3.5],[4.4]]

    dataset = [[1], [1.2], [1.5], [1.8], [2], [3.1], [3.5], [3.9], [5], [5.2], [5.4],
               [1.021], [1.58], [1.51], [10], [10.8], [8.6], [7.7], [1], [1.32], [1.7],
               [8.8], [9], [3.1], [5.5], [3.9], [5.4], [5.22], [5.4], [1.021], [1.58],
               [1.51], [1.8], [8.1], [7.5]]  # , [16.0]

    dataset = np.array(dataset)
    # dataset = np.resize(dataset, (1,4))
    # dataset = np.transpose(dataset)

    MD = DeD(dataset=dataset)
    print(MD)

    # a = mahalanobis(x=dataset, data=dataset)
    # print(a)
    #
    # https://www.machinelearningplus.com/statistics/mahalanobis-distance/
    # https://stackoverflow.com/questions/57475242/implementing-mahalanobis-distance-from-scratch-in-python
    # https://link.springer.com/article/10.1007/s41019-019-0091-y#Tab2
