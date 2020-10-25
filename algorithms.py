import abc
import copy

import matplotlib.pyplot as plt
import functions as fn
import sys
import random
import tkinter
import numpy as np
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure

##
# @brief Abstract class for algorithms
class Algorithm(metaclass=abc.ABCMeta):
    ##
    # @brief Common constructor for all algorithms
    # @param function Test function instance
    # @param pointCloudSize size of generated point clouds
    # @param number of dimensions
    def __init__(self, function : fn.Function, pointCloudSize=10, dimensions=3):
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

class City:
    idCounter = 0
    def __init__(self, coords, name):
        self.coords = coords
        self.name = name
        self.id = City.idCounter
        City.idCounter += 1

    def __eq__(self, other):
        return self.id == other.id

    def print(self):
        print(f'City {self.name}, ID {self.id}, coords {self.coords}')

    @staticmethod
    def generateRandom(bounds):
        coords = []
        dimensions = len(bounds)
        for i in range(dimensions):
            coords.append(np.random.uniform(0, bounds[i]))

        name = chr(City.idCounter + 65)

        return City(coords, name)

class Path:
    def __init__(self, cities, shuffle = True):
        self.cities = copy.deepcopy(cities)
        if(shuffle):
            first = self.getInitialCity()
            otherCities = self.getOtherCitiesList()
            random.shuffle(otherCities)
            self.cities = []
            self.cities.append(first)
            self.cities.extend(otherCities)

    def getPosVector(self):
        posVector = [[],[],[]]

        for city in self.cities:
            posVector[0].append(city.coords[0])
            posVector[1].append(city.coords[1])
            posVector[2].append(city.coords[2])

        posVector[0].append(posVector[0][0])
        posVector[1].append(posVector[1][0])
        posVector[2].append(posVector[2][0])

        return posVector

    def getInitialCity(self):
        return self.cities[0]

    def getOtherCitiesList(self):
        return self.cities[1::]

    def getLength(self):
        length = 0
        localCities = []

        localCities.append(self.getInitialCity())
        localCities.extend(self.getOtherCitiesList())
        localCities.append(self.getInitialCity())

        for i in range(len(localCities) - 1):
            a = np.array(localCities[i].coords)
            b = np.array(localCities[i + 1].coords)
            length += np.linalg.norm(a - b)
        return length

    def print(self):
        print(f'Path length: {self.getLength()}')
        for city in self.cities:
            city.print()
        self.getInitialCity().print()

##
# @brief Genetic algorithm for solving TSP
class TravelingSalesmanGeneticAlgorithm(Algorithm):
    def __init__(self, options):
        super().__init__(None, None)
        self.citiesCount = options['citiesCount'].get()
        self.dimensions = options['dimensions'].get()
        self.populationSize = options['populationSize'].get()
        self.mutationChance = options['mutationChance'].get()
        self.workspaceSize = []

        for i in range(self.dimensions):
            self.workspaceSize.append(options['workspaceSize'][i])

        self.cities = []
        for i in range(self.citiesCount):
            self.cities.append(City.generateRandom(self.workspaceSize))

        self.population = []
        for i in range(self.populationSize):
            self.population.append(Path(self.cities))

        self.bestPath = None

    def reset(self):
        super().reset()

        self.population = []
        for i in range(self.populationSize):
            self.population.append(Path(self.cities))

        self.bestPath = None


    def plot(self, axes):
        axes.clear()
        points = self.population[0].getPosVector()

        axes.scatter(xs=points[0], ys=points[1], zs=points[2], c='g', marker='o', label='cities')

        bestFitness = sys.float_info.max
        bestPath = None
        for path in self.population:
            pathLen = path.getLength()
            if  pathLen < bestFitness:
                bestFitness = pathLen
                bestPath = path

        for path in self.population:
            posVector = path.getPosVector()
            if path is bestPath:
                axes.plot(posVector[0], posVector[1], posVector[2], label=f'fitness: {path.getLength()}', c='r')
            else:
                axes.plot(posVector[0], posVector[1], posVector[2], label=f'fitness: {path.getLength()}', alpha = 0.1)
                pass

        return axes

    def crossover(self, parentA, parentB):
        crossoverAt = np.round(np.random.uniform(1, self.citiesCount))
        offspring = Path([], False)

        for i in range (self.citiesCount):
            if i < crossoverAt:
                offspring.cities.append(parentA.cities[i])

        notInList = []

        for city in parentB.cities:
            if city not in offspring.cities:
                notInList.append(city)

        offspring.cities.extend(notInList)
        return offspring

    def swapPositions(self, list, pos1, pos2):

        list[pos1], list[pos2] = list[pos2], list[pos1]
        return list

    def mutate(self, path):
        if(np.random.uniform(0,1)) < self.mutationChance:
            indexes = random.sample(range(1, self.citiesCount), 2)
            self.swapPositions(list=path.cities, pos1=indexes[0], pos2=indexes[1])


    def solve(self,  ax3d, canvas, maxIterations):
        self.reset()

        print("Solving")
        ##
        # Iterate through algorithm:

        for i in range(maxIterations):
            print(f'iteration {i+1} / {maxIterations}')
            self.solveImpl(currentIterationNumber=i, ax3d=ax3d)
            # Plot each iteration
            ax3d = self.plot(axes=ax3d)

            if canvas.figure.stale:
                canvas.draw_idle()
            canvas.start_event_loop(self.renderDelay)

            self.fitnessHistory.append(self.fitness)
            i += 1
        self.solved = True
        print("Solved")

        self.bestPath.print()

    def solveImpl(self, currentIterationNumber, ax3d=None):
        newPopulation = copy.deepcopy(self.population)
        for j in range(self.citiesCount):
            parentA = self.population[j]

            indexes = list(range(0, self.populationSize))
            indexes.remove(j)
            parentB = self.population[random.choice(indexes)]
            offspring = self.crossover(parentA, parentB)
            self.mutate(offspring)

            if (offspring.getLength() < parentA.getLength()):
                newPopulation[j] = offspring
        self.population = newPopulation

        self.fitness = sys.float_info.max
        self.bestPath = None
        for path in self.population:
            pathLen = path.getLength()
            if pathLen < self.fitness:
                self.fitness = pathLen
                self.bestPath = path



##
# @brief Differential genetic algorithm implementation
class DifferentialGeneticAlgorithm(Algorithm):
    def __init__(self, function : fn.Function, options):
        super().__init__(function, pointCloudSize=options["populationSize"].get(), dimensions=options["dimensions"].get())
        self.scalingFactorF = options['scalingFactorF'].get()
        self.crossoverCR = options['crossoverCR'].get()

        self.pointCloud = self.getRandomPointCloudUniform()

    def reset(self):
        super().reset()
        self.pointCloud = self.getRandomPointCloudUniform()

    def solveImpl(self, currentIterationNumber, ax3d=None):
        newPopulation = copy.deepcopy(self.pointCloud)
        i = 0
        for individual in self.pointCloud:
            indexes = list(range(0, self.pointCloudSize))

            idxR1 = random.choice(indexes)
            r1 = self.pointCloud[idxR1]
            indexes.remove(idxR1)

            idxR2 = random.choice(indexes)
            r2 = self.pointCloud[idxR2]
            indexes.remove(idxR2)

            idxR3 = random.choice(indexes)
            r3 = self.pointCloud[idxR3]
            indexes.remove(idxR3)
            v = []
            u = []

            for j in range(self.dimensions-1):
                elem = (r1[j] - r2[j]) * self.scalingFactorF + r3[j]
                elem = self.clamp(elem, self.function.minimum, self.function.maximum)
                v.append(elem)
                u.append(0)

            j_rnd = np.random.randint(0, self.dimensions-1)

            for j in range(self.dimensions-1):
                if np.random.uniform() < self.crossoverCR or j == j_rnd:
                    u[j] = v[j]
                else:
                    u[j] = individual[j]

            fitness = self.function.getFunctionValue(u)
            parentFitness = self.function.getFunctionValue(individual)

            if(fitness < self.fitness):
                self.fitness = fitness # Only for drawing purposes, best fitness in each generation is selected

            if(fitness <= parentFitness):
                newPopulation[i] = u

            i += 1

        self.pointCloud = newPopulation
