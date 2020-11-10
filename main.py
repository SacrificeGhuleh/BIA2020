import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import traceback

import functions as fn
import algorithms as alg
import numpy as np
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # This import HAS to stay here !!!


class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.initialized = False
        self.master = master
        self.initVars()
        self.createWorkspace()

    def initVars(self):
        # Default padding for GUI elements
        self.defPad = 2

        self.maxIterations = tk.IntVar()
        self.maxIterations.set(100)

        # Selection boxes
        self.functions = ['Sphere', 'Schwefel', 'Rosenbrock', 'Rastrigin', 'Griewank', 'Levy', 'Michalewicz',
                          'Zakharov', 'Ackley']
        self.selectedFunction = tk.StringVar()
        self.selectedFunction.set(self.functions[0])

        self.defaultNotebookFrameWidth = 200
        self.defaultNotebookFrameHeight = 200

        self.renderDelay = tk.DoubleVar()
        self.renderDelay.set(0.05)

    def createWorkspace(self):
        # Frames
        mainFrame = ttk.Frame(self.master)
        mainFrame.grid(row=0, column=0, padx=self.defPad, pady=self.defPad)

        mainConfigFrame = ttk.LabelFrame(mainFrame, text="Config")
        mainConfigFrame.grid(row=0, column=0, padx=self.defPad, pady=self.defPad, sticky=tk.NW + tk.NE)

        commonConfigFrame = ttk.LabelFrame(mainConfigFrame, text="Common")
        commonConfigFrame.grid(row=0, column=0, padx=self.defPad, pady=self.defPad, sticky=tk.NW + tk.NE)

        algoConfigFrame = ttk.LabelFrame(mainConfigFrame, text="Algorithms")
        algoConfigFrame.grid(row=1, column=0, padx=self.defPad, pady=self.defPad, sticky=tk.NW + tk.NE)

        controlsConfigFrame = ttk.LabelFrame(mainConfigFrame, text="Control")
        controlsConfigFrame.grid(row=2, column=0, padx=self.defPad, pady=self.defPad, sticky=tk.NW + tk.NE)

        #
        # Common config
        #

        functionLabel = ttk.Label(commonConfigFrame, text="Function")
        functionLabel.grid(row=0, column=0, padx=self.defPad, pady=self.defPad, sticky=tk.E)

        functionOption = ttk.OptionMenu(commonConfigFrame, self.selectedFunction, None, *self.functions)
        functionOption.grid(row=0, column=1, padx=self.defPad, pady=self.defPad, sticky=tk.W)

        functionOption.bind("<Configure>", self.onSelectFunction)

        iterationLabel = ttk.Label(commonConfigFrame, text="Iterations")
        iterationLabel.grid(row=1, column=0, padx=self.defPad, pady=self.defPad, sticky=tk.E)

        iterationEntry = ttk.Entry(commonConfigFrame, textvariable=self.maxIterations)
        iterationEntry.grid(row=1, column=1, padx=self.defPad, pady=self.defPad, sticky=tk.W)

        renderDelayLabel = ttk.Label(commonConfigFrame, text="Render delay")
        renderDelayLabel.grid(row=2, column=0, padx=self.defPad, pady=self.defPad, sticky=tk.E)

        renderDelaySlider = ttk.Scale(commonConfigFrame, from_=0.0, to=5.0, variable=self.renderDelay)
        renderDelaySlider.grid(row=2, column=1, padx=self.defPad, pady=self.defPad, sticky=tk.W + tk.E)

        renderDelayEntry = ttk.Entry(commonConfigFrame, textvariable=self.renderDelay)
        renderDelayEntry.grid(row=3, column=1, padx=self.defPad, pady=self.defPad, sticky=tk.W + tk.E)

        # style = ttk.Style(root)
        # style.configure('lefttab.TNotebook', tabposition='wn')

        # self.tabsFrame = ttk.Notebook(algoConfigFrame, style='lefttab.TNotebook')
        self.tabsFrame = ttk.Notebook(algoConfigFrame)

        blindFrame = self.createBlindFrame(self.tabsFrame)
        hillClimbFrame = self.createHillClimbFrame(self.tabsFrame)
        annealingFrame = self.createAnnealingFrame(self.tabsFrame)
        tspFrame = self.createTSPFrame(self.tabsFrame)
        dgaFrame = self.createDGAFrame(self.tabsFrame)
        somaFrame = self.createSomaFrame(self.tabsFrame)

        self.tabsFrame.add(blindFrame, text='Blind')
        self.tabsFrame.add(hillClimbFrame, text='Hill Climb')
        self.tabsFrame.add(annealingFrame, text='Annealing')
        self.tabsFrame.add(tspFrame, text='TSP')
        self.tabsFrame.add(dgaFrame, text='Differential GA')
        self.tabsFrame.add(somaFrame, text='SOMA')

        self.tabsFrame.grid(row=0, column=0, padx=self.defPad, pady=self.defPad, sticky=tk.NW + tk.NE)

        runButton = ttk.Button(controlsConfigFrame, text="Run", command=self.run)
        runButton.grid(row=0, column=1, padx=self.defPad, pady=self.defPad, sticky=tk.E)

        graph3dFrame = ttk.LabelFrame(mainFrame, text="3D plot")
        graph3dFrame.grid(row=0, column=1, padx=self.defPad, pady=self.defPad, sticky=tk.NW + tk.NE)

        self.fig3D = Figure(figsize=(5, 4), dpi=100)
        self.canvas3D = FigureCanvasTkAgg(self.fig3D, master=graph3dFrame)
        # self.canvas3D.draw()

        self.graph3Dax = self.fig3D.gca(projection="3d")

        t = np.arange(0, 3, .01)
        self.graph3Dax.plot(t, 2 * np.sin(2 * np.pi * t))

        self.canvas3D.get_tk_widget().grid()

        graph2dFrame = ttk.LabelFrame(mainFrame, text="Fitness history")
        graph2dFrame.grid(row=0, column=2, padx=self.defPad, pady=self.defPad, sticky=tk.NW + tk.NE)

        # self.onSelectFunction(None)
        self.initialized = True

    def onSelectFunction(self, evt):
        if self.initialized is False:
            return
        try:
            func = fn.functionsMap[self.selectedFunction.get()]
            print(f"{func} selected")
            func.plot(axes=self.graph3Dax)
            self.canvas3D.draw()
            self.canvas3D.flush_events()
        except:
            print("Unable to update graph")
            traceback.print_exc()

    def quitApp(self):
        root.destroy()

    def createBlindFrame(self, master):
        frame = ttk.Frame(master, width=self.defaultNotebookFrameWidth, height=self.defaultNotebookFrameHeight)

        self.blindOptions = {
                "pointCloud": tk.IntVar()
        }
        self.blindOptions["pointCloud"].set(60)

        self.getFrameWithEntry(frame, text="Point cloud size", variable=self.blindOptions["pointCloud"]).grid(row=0,
                                                                                                              column=0,
                                                                                                              columnspan=2,
                                                                                                              sticky=tk.E)

        return frame

    def createHillClimbFrame(self, master):
        frame = ttk.Frame(master, width=self.defaultNotebookFrameWidth, height=self.defaultNotebookFrameHeight)

        self.hillClimbOptions = {
                "pointCloud": tk.IntVar(),
                "sigma"     : tk.DoubleVar()
        }
        self.hillClimbOptions["pointCloud"].set(60)
        self.hillClimbOptions["sigma"].set(0.05)

        self.getFrameWithEntry(
                master=frame,
                text="Point cloud size",
                variable=self.hillClimbOptions["pointCloud"]
        ).grid(row=0, column=0, columnspan=2, sticky=tk.E)
        self.getFrameWithSliderAndEntry(
                master=frame,
                text="Sigma",
                variable=self.hillClimbOptions["sigma"],
                from_=0,
                to=1
        ).grid(row=1, column=0, columnspan=2, sticky=tk.E)

        return frame

    def createAnnealingFrame(self, master):
        frame = ttk.Frame(master, width=self.defaultNotebookFrameWidth, height=self.defaultNotebookFrameHeight)

        self.annealingOptions = {
                "pointCloud": tk.IntVar(),
                "temp"      : tk.DoubleVar(),
                "tempMin"   : tk.DoubleVar(),
                "alpha"     : tk.DoubleVar(),
                "sigma"     : tk.DoubleVar(),
                "elitism"   : tk.IntVar(),
                "repeats"   : tk.IntVar(),
        }
        self.annealingOptions["pointCloud"].set(10)
        self.annealingOptions["temp"].set(5000)
        self.annealingOptions["tempMin"].set(0.1)
        self.annealingOptions["alpha"].set(0.99)
        self.annealingOptions["sigma"].set(0.1)
        self.annealingOptions["elitism"].set(True)
        self.annealingOptions["repeats"].set(5)

        self.getFrameWithEntry(
                master=frame,
                text="Point cloud size",
                variable=self.annealingOptions["pointCloud"]
        ).grid(row=0, column=0, columnspan=2, sticky=tk.E)
        self.getFrameWithSliderAndEntry(
                master=frame,
                text="Sigma",
                variable=self.annealingOptions["sigma"],
                from_=0,
                to=1
        ).grid(row=1, column=0, columnspan=2, sticky=tk.E)
        self.getFrameWithSliderAndEntry(
                master=frame,
                text="Temperature",
                variable=self.annealingOptions["temp"],
                from_=1,
                to=10000
        ).grid(row=2, column=0, columnspan=2, sticky=tk.E)
        self.getFrameWithSliderAndEntry(
                master=frame,
                text="Min temperature",
                variable=self.annealingOptions["tempMin"],
                from_=0.0,
                to=1
        ).grid(row=3, column=0, columnspan=2, sticky=tk.E)
        self.getFrameWithSliderAndEntry(
                master=frame,
                text="Alpha",
                variable=self.annealingOptions["alpha"],
                from_=0.01,
                to=0.99
        ).grid(row=4, column=0, columnspan=2, sticky=tk.E)
        self.getFrameWithSliderAndEntry(
                master=frame,
                text="Repeats for T",
                variable=self.annealingOptions["repeats"],
                from_=0.0,
                to=100
        ).grid(row=5, column=0, columnspan=2, sticky=tk.E)
        return frame

    def createTSPFrame(self, master):
        frame = ttk.Frame(master, width=self.defaultNotebookFrameWidth, height=self.defaultNotebookFrameHeight)
        self.tspOptions = {
                'citiesCount'   : tk.IntVar(),
                'dimensions'    : tk.IntVar(),
                'populationSize': tk.IntVar(),
                'workspaceSize' : [10000, 10000, 10000],
                'mutationChance': tk.DoubleVar(),
        }

        self.tspOptions['citiesCount'].set(20)
        self.tspOptions['dimensions'].set(3)
        self.tspOptions['populationSize'].set(20)
        self.tspOptions['mutationChance'].set(0.5)

        self.getFrameWithSliderAndEntry(
                master=frame,
                text="Cities count",
                variable=self.tspOptions["citiesCount"],
                from_=4,
                to=100
        ).grid(row=0, column=0, columnspan=2, sticky=tk.E)

        self.getFrameWithSliderAndEntry(
                master=frame,
                text="Population",
                variable=self.tspOptions["populationSize"],
                from_=1,
                to=100
        ).grid(row=1, column=0, columnspan=2, sticky=tk.E)

        self.getFrameWithSliderAndEntry(
                master=frame,
                text="Mutation chance",
                variable=self.tspOptions["mutationChance"],
                from_=0,
                to=1
        ).grid(row=2, column=0, columnspan=2, sticky=tk.E)

        return frame

    def createDGAFrame(self, master):
        frame = ttk.Frame(master, width=self.defaultNotebookFrameWidth, height=self.defaultNotebookFrameHeight)
        self.dgaOptions = {
                'populationSize': tk.IntVar(),
                'dimensions'    : tk.IntVar(),
                'scalingFactorF': tk.DoubleVar(),
                'crossoverCR'   : tk.DoubleVar(),
        }

        self.dgaOptions['populationSize'].set(20)
        self.dgaOptions['dimensions'].set(3)
        self.dgaOptions['scalingFactorF'].set(0.7)
        self.dgaOptions['crossoverCR'].set(0.7)

        self.getFrameWithSliderAndEntry(
                master=frame,
                text="Population size",
                variable=self.dgaOptions["populationSize"],
                from_=4,
                to=100
        ).grid(row=0, column=0, columnspan=2, sticky=tk.E)

        self.getFrameWithSliderAndEntry(
                master=frame,
                text="Scaling factor F",
                variable=self.dgaOptions["scalingFactorF"],
                from_=0.1,
                to=1.1
        ).grid(row=1, column=0, columnspan=2, sticky=tk.E)

        self.getFrameWithSliderAndEntry(
                master=frame,
                text="Crossover factor CR",
                variable=self.dgaOptions["crossoverCR"],
                from_=0.0,
                to=1.0
        ).grid(row=2, column=0, columnspan=2, sticky=tk.E)

        return frame

    def createSomaFrame(self, master):
        frame = ttk.Frame(master, width=self.defaultNotebookFrameWidth, height=self.defaultNotebookFrameHeight)
        self.somaOptions = {
                'populationSize': tk.IntVar(),
                'dimensions'    : tk.IntVar(),
                'pathLength'    : tk.DoubleVar(),
                'step'          : tk.DoubleVar(),
                'perturbation'  : tk.DoubleVar(),
                'minDiv'        : tk.DoubleVar(),
        }

        self.somaOptions['populationSize'].set(20)
        self.somaOptions['dimensions'].set(3)
        self.somaOptions['pathLength'].set(3)
        self.somaOptions['step'].set(0.11)
        self.somaOptions['perturbation'].set(0.1)
        self.somaOptions['minDiv'].set(-0.1)

        self.getFrameWithSliderAndEntry(
                master=frame,
                text="Population size",
                variable=self.somaOptions["populationSize"],
                from_=10,
                to=100
        ).grid(row=0, column=0, columnspan=2, sticky=tk.E)

        self.getFrameWithSliderAndEntry(
                master=frame,
                text="Path length",
                variable=self.somaOptions["pathLength"],
                from_=1.1,
                to=10
        ).grid(row=1, column=0, columnspan=2, sticky=tk.E)

        self.getFrameWithSliderAndEntry(
                master=frame,
                text="Step",
                variable=self.somaOptions["step"],
                from_=0.11,
                to=self.somaOptions["pathLength"].get()
        ).grid(row=2, column=0, columnspan=2, sticky=tk.E)

        self.getFrameWithSliderAndEntry(
                master=frame,
                text="Perturbation",
                variable=self.somaOptions["perturbation"],
                from_=0,
                to=1
        ).grid(row=3, column=0, columnspan=2, sticky=tk.E)

        self.getFrameWithSliderAndEntry(
                master=frame,
                text="Min Div",
                variable=self.somaOptions["minDiv"],
                from_=1,
                to=100
        ).grid(row=4, column=0, columnspan=2, sticky=tk.E)

        return frame

    def getFrameWithEntry(self, master, text, variable):
        frame = ttk.Frame(master)

        valLabel = ttk.Label(frame, text=text)
        valLabel.grid(row=0, column=0, padx=self.defPad, pady=self.defPad, sticky=tk.E)
        valEntry = ttk.Entry(frame, textvariable=variable)
        valEntry.grid(row=0, column=1, padx=self.defPad, pady=self.defPad, sticky=tk.W + tk.E)

        return frame

    def getFrameWithSliderAndEntry(self, master, text, variable, from_, to):
        frame = ttk.Frame(master)

        valLabel = ttk.Label(frame, text=text)
        valLabel.grid(row=0, column=0, padx=self.defPad, pady=self.defPad, sticky=tk.E)
        valSlider = ttk.Scale(frame, from_=from_, to=to, variable=variable)
        valSlider.grid(row=0, column=1, padx=self.defPad, pady=self.defPad, sticky=tk.W + tk.E)
        valEntry = ttk.Entry(frame, textvariable=variable)
        valEntry.grid(row=1, column=1, padx=self.defPad, pady=self.defPad, sticky=tk.W + tk.E)

        return frame

    def getAlgorithm(self):
        currentTabIdx = self.tabsFrame.index("current")
        func = fn.functionsMap[self.selectedFunction.get()]
        func.clearDict()

        algo = {
                0: alg.BlindAlgorithm(function=func,
                                      pointCloudSize=self.blindOptions["pointCloud"].get()),
                1: alg.HillClimbAlgorithm(function=func,
                                          pointCloudSize=self.hillClimbOptions["pointCloud"].get(),
                                          sigma=self.hillClimbOptions["sigma"].get()),
                2: alg.AnnealingAlgorithm(function=func,
                                          options=self.annealingOptions),
                3: alg.TravelingSalesmanGeneticAlgorithm(options=self.tspOptions),
                4: alg.DifferentialGeneticAlgorithm(function=func, options=self.dgaOptions),
                5: alg.SelfOrganizingMigrationAlgorithm(function=func, options=self.somaOptions),
        }.get(currentTabIdx, None)

        print(f"Func: {func}")
        print(f"Current index: {currentTabIdx}")
        print(f"Alg: {algo}")

        return algo

    def run(self):
        algo = self.getAlgorithm()
        algo.renderDelay = self.renderDelay.get()
        algo.solve(maxIterations=self.maxIterations.get(), ax3d=self.graph3Dax, canvas=self.canvas3D)
        tk.messagebox.showinfo("Done", f"Best found value: {algo.fitness}. For more info check console output.")
        algo.plotFitnessHistory()


if __name__ == '__main__':
    root = tk.Tk()
    root.title("BIA GUI")
    app = Application(master=root)
    root.resizable(False, False)
    app.mainloop()
