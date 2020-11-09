import numpy as np
import abc
import sys

import matplotlib.pyplot as plt
import time


##
# @brief Main abstract class for test functions
class Function(metaclass=abc.ABCMeta):
    ##
    # @brief Common constructor for all test functions
    # @param[in] minimum Minimum value of function
    # @param[in] maximum Maximum value of function
    # @param[in] resolution Resolution, for drawing. Higher resolution creates more accurate graph
    def __init__(self, minimum, maximum, resolution):
        self.minimum = minimum
        self.maximum = maximum
        self.resolution = resolution
        self.bufferedValues = {}

        self.plotX = np.outer(np.linspace(self.minimum, self.maximum, self.resolution), np.ones(self.resolution))
        self.plotY = self.plotX.copy().T
        self.plotZ = self.getFunctionValueImpl((self.plotX, self.plotY))
        self.name = 'Unnamed function'

    ##
    # @brief clear dictionary
    def clearDict(self):
        self.bufferedValues.clear()

    ##
    # @brief Buffered function
    def getFunctionValue(self, x: list):
        tupleX = tuple(x)
        if tupleX not in self.bufferedValues:
            self.bufferedValues[tupleX] = self.getFunctionValueImpl(x)

        return self.bufferedValues[tupleX]

    ##
    # @brief Get minimum coords from vector x
    def getMinimum(self, x: list):
        minCoords = None
        minVal = sys.float_info.max
        for coords in x:
            val = self.getFunctionValue(coords)
            if val < minVal:
                minVal = val
                minCoords = coords
        return minCoords

    ##
    # @brief Abstract function, each function shall be implemented in this function
    @abc.abstractmethod
    def getFunctionValueImpl(self, x):
        pass

    ##
    # @brief Function for plotting each function.
    def plot(self, bestPoint=None, pointsCloud=None, surfaceAlpha=1.0, axes=None):
        ax = None
        if axes is None:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
        else:
            ax = axes
            ax.clear()

        polyc = ax.plot_surface(self.plotX, self.plotY, self.plotZ, cmap=plt.cm.jet, rstride=1, cstride=1, linewidth=1,
                                label=self.name, alpha=surfaceAlpha)
        plt.title(self.name)
        if pointsCloud is not None:
            points = [[], [], []]
            for point in pointsCloud:
                points[0].append(point[0])
                points[1].append(point[1])
                points[2].append(self.getFunctionValue(point))
            ax.scatter(xs=points[0], ys=points[1], zs=points[2], c='g', marker='o', label='considered points')

        if bestPoint is not None:
            funcVal = self.getFunctionValue(bestPoint)
            ax.scatter(xs=bestPoint[0], ys=bestPoint[1], zs=funcVal, c='r', marker='o', label='best point')
        # fix issue with legend
        # https://github.com/matplotlib/matplotlib/issues/4067
        polyc._facecolors2d = polyc._facecolors3d
        polyc._edgecolors2d = polyc._edgecolors3d
        # plt.legend()
        # plt.show()
        return ax


##
# @brief   Sphere function
# @remarks Sphere function is defined as \f$f(x)=\sum_{i=1}^{d}x_i^2\f$
# \n(source <a href="https://www.sfu.ca/~ssurjano/spheref.html">here</a>)
class SphereFunction(Function):
    def __init__(self, minimum, maximum, resolution):
        super().__init__(minimum, maximum, resolution)
        self.name = "Sphere function"

    def getFunctionValueImpl(self, x):
        result = 0.0
        for k in range(0, len(x)):
            result += x[k] ** 2
        return result


##
# @brief   Schwefel function
# @remarks Schwefel function is defined as \f$f(x)=418.9829d - \sum_{i=1}^{d}x_isin(\sqrt{|x_i|})\f$
# \n(source <a href="https://www.sfu.ca/~ssurjano/schwef.html">here</a>)
class SchwefelFunction(Function):
    def __init__(self, minimum, maximum, resolution):
        super().__init__(minimum, maximum, resolution)
        self.name = "Schwefel function"

    def getFunctionValueImpl(self, x):
        result = 0.0
        for i in range(0, len(x)):
            result += x[i] * np.sin(np.sqrt(abs(x[i])))

        result = 418.9829 * len(x) - result
        return result


##
# @brief   Rosenbrock function
# @remarks Rosenbrock function is defined as \f$f(x)=\sum_{i=1}^{d-1} [100(x_{i+1}-x_i^2)^2 + (x_i - 1)^2]\f$
# \n(source <a href="https://www.sfu.ca/~ssurjano/rosen.html">here</a>)
class RosenbrockFunction(Function):
    def __init__(self, minimum, maximum, resolution):
        super().__init__(minimum, maximum, resolution)
        self.name = "Rosenbrock function"

    def getFunctionValueImpl(self, x):
        result = 0.0
        for i in range(0, len(x) - 1):
            result += 100 * ((x[i + 1] - (x[i] ** 2)) ** 2) + ((x[i] - 1) ** 2)
        return result


##
# @brief   Rastrigin function
# @remarks Rastrigin function is defined as \f$f(x)=10d + \sum_{i=1}^{d} [x_i^2 - 10cos(2 \pi x_i)]\f$
# \n(source <a href="https://www.sfu.ca/~ssurjano/rastr.html">here</a>)
class RastriginFunction(Function):
    def __init__(self, minimum, maximum, resolution):
        super().__init__(minimum, maximum, resolution)
        self.name = "Rastrigin function"

    def getFunctionValueImpl(self, x):
        result = 0.0
        for i in range(0, len(x)):
            result += (x[i] ** 2) - 10 * np.cos(2 * np.pi * x[i])
        return 10 * len(x) + result


##
# @brief   Griewank function
# @remarks Griewank function is defined as
# \f$f(x)=\sum_{i=1}^{d} \frac{x_i^2}{4000} - \prod_{i=1}^{d}cos(\frac{x_i}{\sqrt{i}}) + 1  \f$
# \n(source <a href="https://www.sfu.ca/~ssurjano/Griewank.html">here</a>)
class GriewankFunction(Function):
    def __init__(self, minimum, maximum, resolution):
        super().__init__(minimum, maximum, resolution)
        self.name = "Griewank function"

    def getFunctionValueImpl(self, x):
        sm = 0.0
        prod = 1.0
        for i in range(0, len(x)):
            sm += (x[i] ** 2) / 4000
            prod *= np.cos(x[i] / np.sqrt(i + 1))
        return sm - prod + 1


##
# @brief   Levy function
# @remarks Levy function is defined as \f$f(x)= TODO  \f$
# \n(source <a href="https://www.sfu.ca/~ssurjano/levy.html">here</a>)
class LevyFunction(Function):
    def __init__(self, minimum, maximum, resolution):
        super().__init__(minimum, maximum, resolution)
        self.name = "Levy function"

    def getFunctionValueImpl(self, x):
        d = len(x)

        w = []
        for i in range(0, len(x)):
            w.append(1 + (x[i] - 1) / 4)

        term1 = (np.sin(np.pi * w[0])) ** 2
        term3 = (w[d - 1] - 1) ** 2 * (1 + (np.sin(2 * np.pi * w[d - 1])) ** 2)

        sm = 0
        for i in range(0, len(x) - 1):
            wi = w[i]
            sm += (wi - 1) ** 2 * (1 + 10 * (np.sin(np.pi * wi + 1)) ** 2)

        return term1 + sm + term3


##
# @brief   Michalewicz function
# @remarks Michalewicz function is defined as \f$f(x)=-\sum_{i=1}^d \sin(x_i)\sin^{2m}\left(\frac{ix_i^2}{\pi}\right)\f$
# \n(source <a href="https://www.sfu.ca/~ssurjano/michal.html">here</a>)
class MichalewiczFunction(Function):
    def __init__(self, minimum, maximum, resolution):
        super().__init__(minimum, maximum, resolution)
        self.name = "Michalewicz function"

    def getFunctionValueImpl(self, x):
        result = 0
        m = 10
        for i in range(0, len(x)):
            result += np.sin(x[i]) * np.sin(((i + 1) * x[i] ** 2) / np.pi) ** (2 * m)
        return -result


##
# @brief   Zakharov function
# @remarks Zakharov function is defined as \f$f(x) =\sum_{i=1}^n x_i^{2}+(\sum_{i=1}^n 0.5ix_i)^2 + (\sum_{i=1}^n 0.5ix_i)^4 \f$
# \n(source <a href="https://www.sfu.ca/~ssurjano/zakharov.html">here</a>)
class ZakharovFunction(Function):
    def __init__(self, minimum, maximum, resolution):
        super().__init__(minimum, maximum, resolution)
        self.name = "Zakharov function"

    def getFunctionValueImpl(self, x):
        sum1 = 0
        sum2 = 0
        for i in range(0, len(x)):
            xi = x[i]
            sum1 += xi ** 2
            sum2 += 0.5 * (i - 1) * xi
        return sum1 + sum2 ** 2 + sum2 ** 4


##
# @brief   Ackley function
# @remarks Ackley function is defined as \f$f(x) = TODO \f$
# \n(source <a href="https://www.sfu.ca/~ssurjano/ackley.html">here</a>)
class AckleyFunction(Function):
    def __init__(self, minimum, maximum, resolution):
        super().__init__(minimum, maximum, resolution)
        self.name = "Ackley function"

    def getFunctionValueImpl(self, x):
        a = 20
        b = 0.2
        c = 2 * np.pi
        d = len(x)
        sum1 = 0
        sum2 = 0

        for i in range(0, len(x)):
            xi = x[i]
            sum1 += xi ** 2
            sum2 += np.cos(c * xi)

        term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
        term2 = -np.exp(sum2 / d)
        return term1 + term2 + a + np.exp(1)


def test():
    functions = [SphereFunction(-5.12, 5.12, 30),
                 SchwefelFunction(-500, 500, 30),
                 RosenbrockFunction(-5, 10, 30),
                 RastriginFunction(-5.12, 5.12, 30),
                 GriewankFunction(-600, 600, 30),
                 GriewankFunction(-50, 50, 30),
                 GriewankFunction(-5, 5, 30),
                 LevyFunction(-10, 10, 30),
                 MichalewiczFunction(0, np.pi, 30),
                 ZakharovFunction(-5, 10, 30),
                 AckleyFunction(-32.768, 32.768, 60),
                 ]

    ax = None
    plt.show()
    for func in functions:
        ax = func.plot(axes=ax)
        plt.legend()
        plt.pause(1)
        plt.draw()


SphereFunctionInstance = SphereFunction(-5.12, 5.12, 30)
SchwefelFunctionInstance = SchwefelFunction(-500, 500, 30)
RosenbrockFunctionInstance = RosenbrockFunction(-5, 10, 30)
RastriginFunctionInstance = RastriginFunction(-5.12, 5.12, 30)
GriewankFunctionInstance = GriewankFunction(-600, 600, 30)
LevyFunctionInstance = LevyFunction(-10, 10, 30)
MichalewiczFunctionInstance = MichalewiczFunction(0, np.pi, 30)
ZakharovFunctionInstance = ZakharovFunction(-5, 10, 30)
AckleyFunctionInstance = AckleyFunction(-32.768, 32.768, 60)

functionsMap = {
        'Sphere'     : SphereFunctionInstance,
        'Schwefel'   : SchwefelFunctionInstance,
        'Rosenbrock' : RosenbrockFunctionInstance,
        'Rastrigin'  : RastriginFunctionInstance,
        'Griewank'   : GriewankFunctionInstance,
        'Levy'       : LevyFunctionInstance,
        'Michalewicz': MichalewiczFunctionInstance,
        'Zakharov'   : ZakharovFunctionInstance,
        'Ackley'     : AckleyFunctionInstance
}

if __name__ == '__main__':
    test()
