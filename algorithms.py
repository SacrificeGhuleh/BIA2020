import numpy as np
import abc

import matplotlib.pyplot as plt
import functions as fn
import sys
import random


##
# @brief Abstract class for algorithms
class Algorithm(metaclass=abc.ABCMeta):
    def __init__(self, function, dimensions=3):
        if dimensions <= 0:
            raise Exception("dimensions must be unsigned integer number, greater than 0")
        self.function = function
        self.dimensions = dimensions

    ##
    # @brief Abstract function, each algorithm shall be implemented in this function
    @abc.abstractmethod
    def solve(self, maxIterations=-1):
        pass

    ##
    # @brief Generates uniform random point for current function
    def getRandomPointUniform(self):
        randPoint = []
        for i in range(0, self.dimensions - 1):
            randPoint.append(random.uniform(self.function.minimum, self.function.maximum))
        return randPoint

    ##
    # @brief Generates uniformly distributed points
    # @param cloudSize number of generated points in cloud
    def getRandomPointCloudUniform(self, cloudSize):
        points = []
        for i in range(0, cloudSize):
            points.append(self.getRandomPointUniform())
        return points


##
# @brief Blind search algorithm implementation
class BlindAlgorithm(Algorithm):
    def __init__(self, function, pointCloudSize=10, dimensions=3):
        super().__init__(function, dimensions)
        self.solved = False
        self.fitness = sys.float_info.max
        self.bestPoint = None
        self.pointCloudSize = pointCloudSize
        self.fitnessHistory = []
        self.cloudFitnessHistory = []
        self.cloudFitnessHistory = [[], []]

    def reset(self):
        self.solved = False
        self.fitness = sys.float_info.max
        self.bestPoint = None
        self.fitnessHistory = []
        self.cloudFitnessHistory = [[], []]

    ##
    # @brief Implementation of blind search algorithm
    # @param maxIterations maximum number of iterations
    def solve(self, maxIterations=-1):
        if maxIterations <= 0:
            raise Exception("in THIS case maxIterations must be unsigned integer number, greater than 0")

        self.reset()

        ax = None
        plt.show()

        ##
        # Iterate through algorithm:
        for i in range(0, maxIterations):
            ##
            # 1. Generate points in range <function.minimum, function.maximum>
            pointCloud = self.getRandomPointCloudUniform(cloudSize=self.pointCloudSize)

            ##
            # 2. Iterate through points cloud
            for randPoint in pointCloud:
                ##
                # 3. Calculate fitness of each point.
                # If new fitness is better than currently best fitness, overwrite best fitness and save best found point.
                currentFitness = self.function.getFunctionValue(randPoint)
                if currentFitness < self.fitness:
                    self.fitness = currentFitness
                    self.bestPoint = randPoint

                # Save data for ploting later
                self.cloudFitnessHistory[0].append(i)
                self.cloudFitnessHistory[1].append(currentFitness)
            # Plot each iteration
            ax = self.function.plot(pointsCloud=pointCloud, bestPoint=self.bestPoint, surfaceAlpha=0.5, axes=ax)

            plt.legend()
            plt.pause(1)
            plt.draw()

            self.fitnessHistory.append(self.fitness)
        self.solved = True

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


if __name__ == '__main__':
    blindAlg = BlindAlgorithm(function=fn.AckleyFunction(-32.768, 32.768, 60), pointCloudSize=60)
    blindAlg.solve(maxIterations=10)
    print(f'Best found value: {blindAlg.fitness} in point {blindAlg.bestPoint}')
    blindAlg.plotFitnessHistory()
