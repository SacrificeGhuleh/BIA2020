import numpy as np
import abc

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

    ##
    # @brief Abstract function, each function shall be implemented in this function
    @abc.abstractmethod
    def getFunctionValue(self, x):
        pass

    ##
    # @brief Function for plotting each function.
    def plot(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        x = np.outer(np.linspace(self.minimum, self.maximum, self.resolution), np.ones(self.resolution))
        y = x.copy().T
        z = self.getFunctionValue((x, y))

        ax.plot_surface(x, y, z, cmap=plt.cm.jet, rstride=1, cstride=1, linewidth=1, label='Sphere function')
        plt.show()
        return ax


##
# @brief   Sphere function
# @remarks Sphere function is defined as \f$f(x)=\sum_{i=1}^{d}x_i^2\f$
# \n(source <a href="https://www.sfu.ca/~ssurjano/spheref.html">here</a>)
class SphereFunction(Function):
    def __init__(self, minimum, maximum, resolution):
        super().__init__(minimum, maximum, resolution)

    def getFunctionValue(self, x):
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

    def getFunctionValue(self, x):
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

    def getFunctionValue(self, x):
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

    def getFunctionValue(self, x):
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

    def getFunctionValue(self, x):
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

    def getFunctionValue(self, x):
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
# @remarks Michalewicz function is defined as \f$f(x)= TODO  \f$
# \n(source <a href="https://www.sfu.ca/~ssurjano/michal.html">here</a>)
class MichalewiczFunction(Function):
    def __init__(self, minimum, maximum, resolution):
        super().__init__(minimum, maximum, resolution)

    def getFunctionValue(self, x):
        result = 0
        m = 10
        for i in range(0, len(x)):
            result += np.sin(x[i]) * np.sin(((i + 1) * x[i] ** 2) / np.pi) ** (2 * m)
        return -result


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
                 ]

    for func in functions:
        func.plot()
        plt.show()
        time.sleep(0.5)


if __name__ == '__main__':
    test()
