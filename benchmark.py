import sys

import functions as fn
import algorithms as alg
import numpy as np
import xlsxwriter
import xlsxwriter.worksheet
import statistics

maxEvals = 3000

testsCount = 30
dimensions = 30
populationSize = 30

meanRow = testsCount + 2
deviationRow = testsCount + 3
medianRow = testsCount + 4


class VarStub:
    def __init__(self, value) -> None:
        self.val = value

    def get(self):
        return self.val


def getAlgos(function: fn.Function):
    algos = {}

    dgaOptions = {
        'populationSize': VarStub(value=populationSize),
        'dimensions': VarStub(value=dimensions),
        'scalingFactorF': VarStub(value=0.7),
        'crossoverCR': VarStub(value=0.7),
    }

    algos['DGA'] = alg.DifferentialGeneticAlgorithm(function, dgaOptions)

    somaOptions = {
        'populationSize': VarStub(value=populationSize),
        'dimensions': VarStub(value=dimensions),
        'pathLength': VarStub(value=3),
        'step': VarStub(value=0.11),
        'perturbation': VarStub(value=0.1),
        'minDiv': VarStub(value=-0.1),
    }
    algos['SOMA'] = alg.SelfOrganizingMigrationAlgorithm(function, somaOptions)

    fireflyOptions = {
        'dimensions': VarStub(value=dimensions),
        "populationSize": VarStub(value=populationSize),
        'alpha': VarStub(value=0.6),
        'betaAtractivness': VarStub(value=1.0),
    }

    algos['FA'] = alg.FireflyAlgorithm(function, fireflyOptions)

    tlbaOptions = {
        'dimensions': VarStub(value=3),
        "populationSize": VarStub(value=20),
    }
    algos['TLBA'] = alg.TeachingLearningBasedAlgorithm(function, tlbaOptions)
    return algos


def populateWorkbook(sheet: xlsxwriter.worksheet.Worksheet):
    sheet.set_column(0, 0, 15)

    for i in range(1, testsCount + 1):
        sheet.write(i + 1, 0, f"Experiment {i}")

    sheet.write(meanRow, 0, "Mean")
    sheet.write(deviationRow, 0, "Deviation")
    sheet.write(medianRow, 0, "Median")

    referenceCol = 1
    for i in range(len(fn.functionsMap)):
        key = list(fn.functionsMap.keys())[i]

        function = fn.functionsMap[key]
        function.maxEval = maxEvals
        function.curEval = 0
        function.clearDict()

        algorithms = getAlgos(function)
        sheet.merge_range(0, referenceCol, 0, referenceCol + len(algorithms) - 1, key)
        for j in range(len(algorithms)):
            algoKey = list(algorithms.keys())[j]
            alg = algorithms[algoKey]
            sheet.write(1, referenceCol + j, algoKey)
            fitnessResults = []
            for testIdx in range(testsCount):
                try:
                    alg.reset()
                    alg.solve(None, None, maxEvals * 2)
                except AssertionError as err:
                    print(f"Assertion failed: {sys.exc_info()}")
                    print(f"Current algo: {algoKey}")
                    raise
                except:
                    # Exception is thrown, when max evals is exceeded. This is used to stop algorithm immediately
                    pass
                # sheet.write(2 + testIdx, referenceCol + j, testIdx)
                sheet.write(2 + testIdx, referenceCol + j, alg.fitness)
                fitnessResults.append(alg.fitness)
            sheet.write(meanRow, referenceCol + j, statistics.mean(fitnessResults))
            sheet.write(deviationRow, referenceCol + j, statistics.stdev(fitnessResults))
            sheet.write(medianRow, referenceCol + j, statistics.median(fitnessResults))
        referenceCol += len(algorithms)


if __name__ == '__main__':
    # Create an new Excel file and add a worksheet.
    workbook = xlsxwriter.Workbook('BIA2020_zvo0016.xlsx')
    worksheet = workbook.add_worksheet()

    try:
        populateWorkbook(worksheet)
    except:
        pass
    finally:
        workbook.close()
