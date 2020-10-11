import numpy as np
import abc

import matplotlib.pyplot as plt
import functions as fn
import sys
import random


##
# @brief Abstract class for algorithms
class Algorithm(metaclass=abc.ABCMeta):
    ##
    # @brief Common constructor for all algorithms
    # @param function Test function instance
    # @param pointCloudSize size of generated point clouds
    # @param number of dimensions
    def __init__(self, function, pointCloudSize=10, dimensions=3):
        if dimensions <= 0:
            raise Exception("dimensions must be unsigned integer number, greater than 0")
        self.function = function
        self.dimensions = dimensions
        self.pointCloudSize = pointCloudSize

        self.solved = False
        self.fitness = sys.float_info.max
        self.bestPoint = None
        self.pointCloud = None
        self.fitnessHistory = []
        self.cloudFitnessHistory = [[], []]
        self.renderDelay = 0.5

    def reset(self):
        self.solved = False
        self.fitness = sys.float_info.max
        self.bestPoint = None
        self.pointCloud = None
        self.fitnessHistory = []
        self.cloudFitnessHistory = [[], []]

    ##
    # @brief Main function for solving
    # @remarks This function implements most common code for all algorithms (e.g. drawing)
    # @param maxIterations maximum number of iterations
    def solve(self, maxIterations=-1):
        self.reset()

        ax = None
        plt.show()
        print("Solving")
        ##
        # Iterate through algorithm:
        for i in range(0, maxIterations):
            print(f"  iteration: {i}")
            self.solveImpl(currentIterationNumber=i, ax=ax)

            # Plot each iteration
            ax = self.function.plot(pointsCloud=self.pointCloud, bestPoint=self.bestPoint, surfaceAlpha=0.5, axes=ax)

            plt.legend()
            plt.pause(self.renderDelay)
            plt.draw()

            self.fitnessHistory.append(self.fitness)
        self.solved = True
        print("Solved")

    ##
    # @brief Abstract function, each algorithm shall be implemented in this function
    # @remarks This function shall be the heart of each algorithm. This function is called each iteration.
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
    def getRandomPointCloudUniform(self, minimum=None, maximum=None):
        if minimum is None:
            minimum = self.function.minimum
        if maximum is None:
            maximum = self.function.maximum

        points = []
        for i in range(0, self.pointCloudSize):
            points.append(self.getRandomPointUniform(minimum, maximum))
        return points

    ##
    # @brief Generates normally distributed point. Points are clamped to be always in domain of used function.
    def getRandomPointNormal(self, point, sigma):
        randPoint = []
        for i in range(0, self.dimensions - 1):
            randPoint.append(
                    self.clamp(np.random.normal(point[i], sigma), self.function.minimum, self.function.maximum))
        return randPoint

    ##
    # @brief Generates normally distributed point cloud.
    def getRandomPointCloudNormal(self, point, sigma, cloudSize):
        points = []
        for i in range(0, cloudSize):
            points.append(self.getRandomPointNormal(point, sigma))
        return points

    ##
    # @clamp value to be present in defined range
    # @param num value to be clamped
    # @param lowerBound lower bound of defined interval
    # @param upperBound upper bound of defined interval
    def clamp(self, num, lowerBound, upperBound):
        return max(lowerBound, min(num, upperBound))


##
# @brief Blind search algorithm implementation
class BlindAlgorithm(Algorithm):
    def __init__(self, function, pointCloudSize=10, dimensions=3):
        super().__init__(function, pointCloudSize, dimensions)

    def solveImpl(self, currentIterationNumber, ax=None):
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


##
# @brief Hill climb algorithm implementation
class HillClimbAlgorithm(Algorithm):
    ##
    # @brief Constructor for Hill climb algorithm
    # @param sigma Range for generating random points. Relative to defined domain.
    def __init__(self, function, pointCloudSize=10, dimensions=3, sigma=0.1):
        super().__init__(function, pointCloudSize, dimensions)

        self.sigma = sigma * np.abs(function.maximum - function.minimum)
        self.bestPoint = self.getRandomPointUniform()

    def reset(self):
        super().reset()
        self.bestPoint = self.getRandomPointUniform()

    def solveImpl(self, currentIterationNumber, ax=None):
        ##
        # 1. Generate normally distributed random point across domain
        self.pointCloud = self.getRandomPointCloudNormal(self.bestPoint, self.sigma, cloudSize=self.pointCloudSize)
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


##
# @brief Blind search algorithm implementation
class AnnealingAlgorithm(Algorithm):
    def __init__(self, function, pointCloudSize=10, dimensions=3):
        super().__init__(function, pointCloudSize, dimensions)

    def solveImpl(self, currentIterationNumber, ax=None):
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


if __name__ == '__main__':
    # alg = HillClimbAlgorithm(function=fn.AckleyFunction(-32.768, 32.768, 60), pointCloudSize=60)
    # alg = HillClimbAlgorithm(function=fn.SphereFunctionInstance, pointCloudSize=60, sigma=0.05)
    alg = HillClimbAlgorithm(function=fn.AckleyFunctionInstance, pointCloudSize=60, sigma=0.05)
    alg.solve(maxIterations=30)
    print(f'Best found value: {alg.fitness} in point {alg.bestPoint}')
    alg.plotFitnessHistory()
