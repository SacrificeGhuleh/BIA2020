import numpy as np
import abc

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import axes3d
import time


class Function(metaclass=abc.ABCMeta):
    def __init__(self, minimum, maximum, resolution):
        self.minimum = minimum
        self.maximum = maximum
        self.resolution = resolution

    @abc.abstractmethod
    def getFunctionValue(self, vector):
        pass

    def plot(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        x = np.outer(np.linspace(self.minimum, self.maximum, self.resolution), np.ones(self.resolution))
        y = x.copy().T
        z = self.getFunctionValue((x, y))

        ax.plot_surface(x, y, z, cmap=plt.cm.jet, rstride=1, cstride=1, linewidth=1, label='Sphere function')
        plt.show()
        return ax


class SphereFunction(Function):
    def __init__(self, minimum, maximum, resolution):
        super().__init__(minimum, maximum, resolution)

    def getFunctionValue(self, vector):
        result = 0.0
        for k in range(0, len(vector)):
            result += vector[k] * vector[k]
        return result


class SchwefelFunction(Function):
    def __init__(self, minimum, maximum, resolution):
        super().__init__(minimum, maximum, resolution)

    def getFunctionValue(self, vector):
        result = 0.0
        for i in range(0, len(vector)):
            result += vector[i] * np.sin(np.sqrt(abs(vector[i])))

        result = 418.9829 * len(vector) - result
        return result


class RosenbrockFunction(Function):
    def __init__(self, minimum, maximum, resolution):
        super().__init__(minimum, maximum, resolution)

    def getFunctionValue(self, vector):
        result = 0.0
        for i in range(0, len(vector) - 1):
            temp = 100 * pow((vector[i + 1] - pow(vector[i], 2)), 2) + pow((1 - vector[i]), 2)
            result += temp
        return result


class RastriginFunction(Function):
    def __init__(self, minimum, maximum, resolution):
        super().__init__(minimum, maximum, resolution)

    def getFunctionValue(self, vector):
        result = 0.0
        for i in range(0, len(vector)):
            temp = pow(vector[i], 2) - 10 * np.cos(2 * np.pi * vector[i])
            result += temp
        return 10 * len(vector) + result


def test():
    functions = [SphereFunction(-5.12, 5.12, 30), SchwefelFunction(-500, 500, 30), RosenbrockFunction(-5, 10, 30),
             RastriginFunction(-5.12, 5.12, 30)]

    for func in functions:
        func.plot()
        plt.show()
        time.sleep(0.5)

if __name__ == '__main__':
    test()
