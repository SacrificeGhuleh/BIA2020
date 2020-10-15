import time

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
        self.renderDelay = 0.1

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
    def solve(self, ax3d, canvas, maxIterations):
        self.reset()

        print("Solving")
        ##
        # Iterate through algorithm:
        for i in range(0, maxIterations):
            print(f"  iteration: {i}")
            self.solveImpl(currentIterationNumber=i, ax3d=ax3d)

            # Plot each iteration
            ax3d = self.function.plot(pointsCloud=self.pointCloud, bestPoint=self.bestPoint, surfaceAlpha=0.5, axes=ax3d)

            if canvas.figure.stale:
                canvas.draw_idle()
            canvas.start_event_loop(self.renderDelay)

            self.fitnessHistory.append(self.fitness)
        self.solved = True
        print("Solved")

    ##
    # @brief Abstract function, each algorithm shall be implemented in this function
    # @remarks This function shall be the heart of each algorithm. This function is called each iteration.
    @abc.abstractmethod
    def solveImpl(self, currentIterationNumber, ax3d=None):
        pass

    ##
    # @brief Plot history of fitness
    # @remarks Plot shows change of fitness value and all fitness values from generated points
    def plotFitnessHistory(self):
        if not self.solved:
            raise Exception("Algorithm is not solved, unable to plot graph")
        print("Ploting fitness history...")
        plt.plot(self.fitnessHistory, label='fitness history')

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
    def __init__(self, function, pointCloudSize=10):
        super().__init__(function, pointCloudSize)

    def solveImpl(self, currentIterationNumber, ax3d=None):
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
    def __init__(self, function, pointCloudSize=10, sigma=0.1):
        super().__init__(function, pointCloudSize)

        self.sigma = sigma * np.abs(function.maximum - function.minimum)
        self.bestPoint = self.getRandomPointUniform()

    def reset(self):
        super().reset()
        self.bestPoint = self.getRandomPointUniform()

    def solveImpl(self, currentIterationNumber, ax3d=None):
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
    def __init__(self, function, options):
        super().__init__(function, pointCloudSize=options["pointCloud"].get())
        self.temp = options["temp"].get()
        self.tempMin = options["tempMin"].get()
        self.alpha = options["alpha"].get()
        self.sigma = options["sigma"].get() * np.abs(function.maximum - function.minimum)
        self.elitism = options["elitism"].get()
        self.repeatsForTemperature = options["repeats"].get()
        self.curTemp = self.temp

    def reset(self):
        super().reset()
        self.curTemp = self.temp
        self.bestPoint = self.getRandomPointUniform()
        self.fitness = self.function.getFunctionValue(self.bestPoint)

    def solveImpl(self, currentIterationNumber, ax3d=None):
        ##
        # 1. Generate normally distributed random point across domain
        self.pointCloud = self.getRandomPointCloudNormal(self.bestPoint, self.sigma, cloudSize=self.pointCloudSize)
        for repeat in range(self.repeatsForTemperature):
            print(f"  iteration: {currentIterationNumber}, temperature: {self.curTemp} / minimal temperature: {self.tempMin}")
            ##
            # randomly select point from the set of neighbors
            neighbor = random.choice(self.pointCloud)
            neighborFitness = self.function.getFunctionValue(neighbor)
            delta = neighborFitness - self.fitness
            if delta < 0:
                self.bestPoint = neighbor
                self.fitness = neighborFitness
            else:
                r = np.random.uniform(0, 1)
                if r < np.exp(-delta / self.curTemp):
                    self.bestPoint = neighbor
                    self.fitness = neighborFitness
        self.curTemp = self.alpha * self.curTemp

    def solve(self,  ax3d, canvas, maxIterations):
        self.reset()

        print("Solving")
        i = 0
        ##
        # Iterate through algorithm:
        while self.curTemp > self.tempMin:
            self.solveImpl(currentIterationNumber=i, ax3d=ax3d)

            # Plot each iteration
            ax3d = self.function.plot(pointsCloud=self.pointCloud, bestPoint=self.bestPoint, surfaceAlpha=0.5, axes=ax3d)

            if canvas.figure.stale:
                canvas.draw_idle()
            canvas.start_event_loop(self.renderDelay)

            self.fitnessHistory.append(self.fitness)
            i += 1
        self.solved = True
        print("Solved")

##
# @brief Genetic algorithm for solving TSP
class TravelingSalesmanGeneticAlgorithm(Algorithm):
    def __init__(self, function, options):
        self.citiesCount = options['citiesCount']
        self.dimensions = options['dimensions']
        self.workspaceSize = []

        for i in range(self.dimensions):
            self.workspaceSize[i] = options['workspaceSize'][i]

        self.cities = []
        for i in range(self.citiesCount):
            self.cities.append([])


    def reset(self):
        super().reset()

    
    def getRandomCities(self):
        for i in range(self.citiesCount):
            city = []
            for j in range(self.dimensions):
                city[j] = np.random.uniform(0, self.workspaceSize[j])
            self.cities[i] = city

