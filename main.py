import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import functions as fn
import algorithms as alg


class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.initVars()
        self.createWorkspace()

    def initVars(self):
        # Default padding for GUI elements
        self.defPad = 2

        self.maxIterations = tk.IntVar()
        self.maxIterations.set(1000)

        # Selection boxes
        self.functions = ['Sphere', 'Schwefel', 'Rosenbrock', 'Rastrigin', 'Griewank', 'Levy', 'Michalewicz',
                          'Zakharov', 'Ackley']
        self.selectedFunction = tk.StringVar()
        self.selectedFunction.set(self.functions[0])

        self.defaultNotebookFrameWidth = 200
        self.defaultNotebookFrameHeight = 200

    def createWorkspace(self):
        # Frames
        mainConfigFrame = ttk.LabelFrame(root, text="Config")
        mainConfigFrame.grid(row=0, column=0, padx=self.defPad, pady=self.defPad, sticky=tk.NW + tk.NE)

        commonConfigFrame = ttk.LabelFrame(mainConfigFrame, text="Common")
        commonConfigFrame.grid(row=0, column=0, padx=self.defPad, pady=self.defPad, sticky=tk.NW + tk.NE)

        algoConfigFrame = ttk.LabelFrame(mainConfigFrame, text="Algorithms")
        algoConfigFrame.grid(row=1, column=0, padx=self.defPad, pady=self.defPad, sticky=tk.NW + tk.NE)

        controlsConfigFrame = ttk.LabelFrame(mainConfigFrame, text="Control")
        controlsConfigFrame.grid(row=2, column=0, padx=self.defPad, pady=self.defPad, sticky=tk.NW + tk.NE)

        functionLabel = ttk.Label(commonConfigFrame, text="Function")
        functionLabel.grid(row=0, column=0, padx=self.defPad, pady=self.defPad, sticky=tk.E)

        functionOption = ttk.OptionMenu(commonConfigFrame, self.selectedFunction, *self.functions)
        functionOption.grid(row=0, column=1, padx=self.defPad, pady=self.defPad, sticky=tk.W)

        iterationLabel = ttk.Label(commonConfigFrame, text="Iterations")
        iterationLabel.grid(row=1, column=0, padx=self.defPad, pady=self.defPad, sticky=tk.E)

        iterationEntry = ttk.Entry(commonConfigFrame, textvariable=self.maxIterations)
        iterationEntry.grid(row=1, column=1, padx=self.defPad, pady=self.defPad, sticky=tk.W)

        # style = ttk.Style(root)
        # style.configure('lefttab.TNotebook', tabposition='wn')

        # self.tabsFrame = ttk.Notebook(algoConfigFrame, style='lefttab.TNotebook')
        self.tabsFrame = ttk.Notebook(algoConfigFrame)

        blindFrame = self.createBlindFrame(self.tabsFrame)
        hillClimbFrame = self.createHillClimbFrame(self.tabsFrame)
        annealingFrame = self.createAnnealingFrame(self.tabsFrame)

        self.tabsFrame.add(blindFrame, text='Blind')
        self.tabsFrame.add(hillClimbFrame, text='Hill Climb')
        self.tabsFrame.add(annealingFrame, text='Annealing')

        self.tabsFrame.grid(row=0, column=0, padx=self.defPad, pady=self.defPad, sticky=tk.NW + tk.NE)

        runButton = ttk.Button(controlsConfigFrame, text="Run", command=self.run)
        runButton.grid(row=0, column=1, padx=self.defPad, pady=self.defPad, sticky=tk.E)

    def quitApp(self):
        root.destroy()

    def createBlindFrame(self, master):
        frame = ttk.Frame(master, width=self.defaultNotebookFrameWidth, height=self.defaultNotebookFrameHeight)

        self.blindOptions = {
                "pointCloud": tk.IntVar()
        }
        self.blindOptions["pointCloud"].set(60)

        pointCloudSizeLabel = ttk.Label(frame, text="Point cloud size")
        pointCloudSizeLabel.grid(row=0, column=0, padx=self.defPad, pady=self.defPad, sticky=tk.E)

        pointCloudSizeSpin = ttk.Entry(frame, textvariable=self.blindOptions["pointCloud"])
        pointCloudSizeSpin.grid(row=0, column=1, padx=self.defPad, pady=self.defPad, sticky=tk.W)

        return frame

    def createHillClimbFrame(self, master):
        frame = ttk.Frame(master, width=self.defaultNotebookFrameWidth, height=self.defaultNotebookFrameHeight)

        self.hillClimbOptions = {
                "pointCloud": tk.IntVar(),
                "sigma"     : tk.DoubleVar()
        }
        self.hillClimbOptions["pointCloud"].set(60)
        self.hillClimbOptions["sigma"].set(0.05)

        pointCloudSizeLabel = ttk.Label(frame, text="Point cloud size")
        pointCloudSizeLabel.grid(row=0, column=0, padx=self.defPad, pady=self.defPad, sticky=tk.E)

        pointCloudSizeEntry = ttk.Entry(frame, textvariable=self.hillClimbOptions["pointCloud"])
        pointCloudSizeEntry.grid(row=0, column=1, padx=self.defPad, pady=self.defPad, sticky=tk.W + tk.E)

        sigmaLabel = ttk.Label(frame, text="Sigma")
        sigmaLabel.grid(row=1, column=0, padx=self.defPad, pady=self.defPad, sticky=tk.E)

        sigmaSlider = ttk.Scale(frame, from_=0.0, to=1.0, variable=self.hillClimbOptions["sigma"])
        sigmaSlider.grid(row=1, column=1, padx=self.defPad, pady=self.defPad, sticky=tk.W + tk.E)

        sigmaEntry = ttk.Entry(frame, textvariable=self.hillClimbOptions["sigma"])
        sigmaEntry.grid(row=2, column=1, padx=self.defPad, pady=self.defPad, sticky=tk.W + tk.E)

        return frame

    def createAnnealingFrame(self, master):
        frame = ttk.Frame(master, width=self.defaultNotebookFrameWidth, height=self.defaultNotebookFrameHeight)

        self.annealingOptions = {
                "pointCloud": tk.IntVar()
        }
        self.annealingOptions["pointCloud"].set(60)

        return frame

    def getAlgorithm(self):
        currentTabIdx = self.tabsFrame.index("current")
        func = fn.functionsMap[self.selectedFunction.get()]

        algo = {
                0: alg.BlindAlgorithm(function=func,
                                      pointCloudSize=self.blindOptions["pointCloud"].get()),
                1: alg.HillClimbAlgorithm(function=func,
                                          pointCloudSize=self.hillClimbOptions["pointCloud"].get(),
                                          sigma=self.hillClimbOptions["sigma"].get()),
                2: alg.AnnealingAlgorithm(function=func,
                                          pointCloudSize=self.annealingOptions["pointCloud"].get()),
        }.get(currentTabIdx, None)

        print(f"Func: {func}")
        print(f"Current index: {currentTabIdx}")
        print(f"Alg: {algo}")
        return algo

    def run(self):
        algo = self.getAlgorithm()
        algo.solve(maxIterations=self.maxIterations.get())
        tk.messagebox.showinfo("Done", f"Best found value: {algo.fitness} in point {algo.bestPoint}")
        print(f'Best found value: {algo.fitness} in point {algo.bestPoint}')
        algo.plotFitnessHistory()

if __name__ == '__main__':
    root = tk.Tk()
    root.title("BIA GUI")
    app = Application(master=root)
    root.resizable(False, False)
    app.mainloop()
