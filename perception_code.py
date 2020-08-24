import numpy as np
import matplotlib.pyplot as plt
import imageio
import os


def imageToGif(inputName, outfileName):
    files = os.listdir(inputName)
    print(files)
    frames = []
    for file in files:
        frames.append(imageio.imread(inputName + '\\' + file))
    imageio.mimsave(outfileName, frames, 'GIF', duration=0.01)


def make_point(point_number, dim, scale):
    """
    生成分类点
    :param point_number: 点的数目（int)
    :param dim: 点的维数(int)
    :param scale: 点的范围(int)
    :return:
    """
    # np.random.seed(10)
    X = np.random.random([point_number, dim]) * scale
    Y = np.zeros(point_number)
    sum_X = np.sum(X, axis=1)
    for index in range(point_number):
        if sum_X[index] - scale < 0:
            Y[index] = -1
        else:
            Y[index] = 1
    return X, Y


class Plotting(object):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def open_in(self):
        plt.ion()

    def close(self):
        plt.ioff()
        plt.show()

    def vis_plot(self, weight, b, number):
        plt.cla()
        plt.xlim(0, np.max(self.X.T[0]) + 1)
        plt.ylim(0, np.max(self.X.T[1]) + 1)
        plt.scatter(self.X.T[0], self.X.T[1], c=self.Y)
        if True in list(weight == 0):
            plt.plot(0, 0)
        else:
            x1 = -b / weight[0]
            x2 = -b / weight[1]
            plt.plot([x1, 0], [0, x2])
        plt.title('change time:%d' % number)
        number1 = "%05d"%number
        if number > 450:
            plt.savefig(r'pil\%s.png' % number1)
        plt.pause(0.01)

    def just_plot_result(self, weight, b):
        plt.scatter(self.X.T[0], self.X.T[1], c=self.Y)
        x1 = -b / weight[0]
        x2 = -b / weight[1]
        plt.plot([x1, 0], [0, x2])
        plt.show()


class PerceptionMethod(object):
    def __init__(self, X, Y, eta):
        if X.shape[0] != Y.shape[0]:
            raise ValueError('Error,X and Y must be same when axis=0 ')
        else:
            self.X = X
            self.Y = Y
            self.eta = eta

    def ini_Per(self):
        weight = np.zeros(self.X.shape[1])
        b = 0
        number = 0
        mistake = True
        while mistake is True:
            mistake = False
            for index in range(self.X.shape[0]):
                if self.Y[index] * (weight @ self.X[index] + b) <= 0:
                    weight += self.eta * self.Y[index] * self.X[index]
                    b += self.eta * self.Y[index]
                    number += 1
                    print(weight, b)
                    mistake = True
                    break
        return weight, b

    def plot_ini_Per(self):
        if self.X.shape[1] != 2:
            raise ValueError("dimension doesn't support")
        else:
            weight = np.zeros(self.X.shape[1])
            b = 0
            number = 0
            mistake = True
            Vis = Plotting(self.X, self.Y)
            while mistake is True:
                mistake = False
                Vis.open_in()
                Vis.vis_plot(weight, b, number)
                for index in range(self.X.shape[0]):
                    if self.Y[index] * (weight @ self.X[index] + b) <= 0:
                        weight += self.eta * self.Y[index] * self.X[index]
                        b += self.eta * self.Y[index]
                        number += 1
                        print('error time:', number)
                        print(weight, b)
                        mistake = True
                        break
            Vis.close()
        return weight, b

    def dual_Per(self):
        Gram = np.dot(self.X, self.X.T)
        alpha = np.zeros(self.X.shape[0])
        b = 0
        mistake = True
        while mistake is True:
            mistake = False
            for index in range(self.X.shape[0]):
                if self.Y[index] * (alpha * self.Y @ Gram[index] + b) <= 0:
                    alpha[index] += self.eta
                    b += self.eta * self.Y[index]
                    print(alpha, b)
                    mistake = True
                    break
        weight = self.Y * alpha @ self.X
        return weight, b


if __name__ == '__main__':
    imageToGif(r'D:\py文件\zhihu_code\machine_learning\pil', 'my1.GIF')
    # X = np.array([[3, 3], [4, 3], [1, 1]])
    # Y = np.array([1, 1, -1])
    ##############################################################
    X, Y = make_point(15, 2, 10)
    PER = PerceptionMethod(X, Y, 1)
    weight, b = PER.plot_ini_Per()
    #############################################################
    # vis = Plotting(X, Y)
    # vis.just_plot_result(weight, b)
    # dual_perception(X, Y, 1)
