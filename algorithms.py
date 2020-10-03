import numpy as np
import abc

import matplotlib.pyplot as plt
import functions as fn
import sys
import random


##
# @brief Abstract class for algorithms
class Algorithm(metaclass=abc.ABCMeta):
    def __init__(self, function, pointCloudSize=10, dimensions=3):
        if dimensions <= 0:
            raise Exception("dimensions must be unsigned integer number, greater than 0")
        self.function = function
        self.dimensions = dimensions
        self.solved = False
        self.fitness = sys.float_info.max
        self.bestPoint = None
        self.pointCloud = None
        self.fitnessHistory = []
        self.cloudFitnessHistory = [[], []]
        self.pointCloudSize = pointCloudSize

    def reset(self):
        self.solved = False
        self.fitness = sys.float_info.max
        self.bestPoint = None
        self.fitnessHistory = []
        self.cloudFitnessHistory = [[], []]

    ##
    # @brief Abstract function, each algorithm shall be implemented in this function
    def solve(self, maxIterations=-1, ax=None):
        self.reset()

        ax = None
        plt.show()

        ##
        # Iterate through algorithm:
        for i in range(0, maxIterations):
            self.solveImpl(currentIterationNumber=i, ax=ax)

            # Plot each iteration
            ax = self.function.plot(pointsCloud=self.pointCloud, bestPoint=self.bestPoint, surfaceAlpha=0.5, axes=ax)

            plt.legend()
            plt.pause(1)
            plt.draw()

            self.fitnessHistory.append(self.fitness)
        self.solved = True

    ##
    # @brief Abstract function, each algorithm shall be implemented in this function
    @abc.abstractmethod
    def solveImpl(self, currentIterationNumber, ax=None):
        pass

    ##
    # @brief Plot history of fitness
    # @remarks Plot shows change of fitness value and all fitness values from generated points
    def plotFitnessHistory(self):
        if not self.solved:
            raise Exception("Algorithm is not solved, unable to plot graph")
        print("Ploting fitness history...")
        plt.show()
        plt.plot(self.fitnessHistory, label='fitness history')
        plt.scatter(self.cloudFitnessHistory[0], self.cloudFitnessHistory[1], c='g',
                    label='considered fitness\nin generation')
        plt.ylabel("fitness")
        plt.xlabel("iteration")
        plt.title('fitness history')
        plt.legend(bbox_to_anchor=(1, 0), loc='lower left', fontsize='small')
        plt.draw()
        plt.show()

    ##
    # @brief Generates uniform random point for current function
    def getRandomPointUniform(self, minimum=None, maximum=None):
        if minimum is None:
            minimum = self.function.minimum
        if maximum is None:
            maximum = self.function.maximum

        randPoint = []
        for i in range(0, self.dimensions - 1):
            randPoint.append(np.random.uniform(minimum, maximum))
        return randPoint

    ##
    # @brief Generates uniformly distributed points
    # @param cloudSize number of generated points in cloud
    def getRandomPointCloudUniform(self, minimum=None, maximum=None):
        if minimum is None:
            minimum = self.function.minimum
        if maximum is None:
            maximum = self.function.maximum

        points = []
        for i in range(0, self.pointCloudSize):
            points.append(self.getRandomPointUniform(minimum, maximum))
        return points

    def getRandomPointNormal(self, point, sigma):
        randPoint = []
        for i in range(0, self.dimensions - 1):
            randPoint.append(np.random.normal(point[i], sigma))
        return randPoint

    def getRandomPointCloudNormal(self, point, sigma, cloudSize):
        points = []
        for i in range(0, cloudSize):
            points.append(self.getRandomPointNormal(point, sigma))
        return points

##
# @brief Blind search algorithm implementation
class BlindAlgorithm(Algorithm):
    def __init__(self, function, pointCloudSize=10, dimensions=3):
        super().__init__(function, pointCloudSize, dimensions)

    def solveImpl(self, currentIterationNumber , ax=None):
        ##
        # 1. Generate uniformly distributed random point across domain
        self.pointCloud = self.getRandomPointCloudUniform()
        ##
        # 2. Iterate through points cloud
        for randPoint in self.pointCloud:
            ##
            # 3. Calculate fitness of each point.
            # If new fitness is better than currently best fitness, overwrite best fitness and save best found point.
            currentFitness = self.function.getFunctionValue(randPoint)
            if currentFitness < self.fitness:
                self.fitness = currentFitness
                self.bestPoint = randPoint

            # Save data for ploting later
            self.cloudFitnessHistory[0].append(currentIterationNumber)
            self.cloudFitnessHistory[1].append(currentFitness)


class HillClimbAlgorithm(Algorithm):
    def __init__(self, function, pointCloudSize=10, dimensions=3, sigma=5):
        super().__init__(function, pointCloudSize, dimensions)
        self.pointCloudSize = pointCloudSize
        self.sigma = sigma

if __name__ == '__main__':
    alg = BlindAlgorithm(function=fn.AckleyFunctionInstance, pointCloudSize=60)
    # alg = HillClimbAlgorithm(function=fn.AckleyFunction(-32.768, 32.768, 60), pointCloudSize=60)
    # alg = HillClimbAlgorithm(function=fn.sphereFunctionInstance, pointCloudSize=60)
    alg.solve(maxIterations=10)
    print(f'Best found value: {alg.fitness} in point {alg.bestPoint}')
    alg.plotFitnessHistory()
