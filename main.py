import tkinter as tk
from tkinter import ttk


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

        # tabsFrame = ttk.Notebook(algoConfigFrame, style='lefttab.TNotebook')
        tabsFrame = ttk.Notebook(algoConfigFrame)

        blindFrame = self.createBlindFrame(tabsFrame)
        hillClimbFrame = self.createHillClimbFrame(tabsFrame)
        annealingFrame = self.createAnnealingFrame(tabsFrame)

        tabsFrame.add(blindFrame, text='Blind')
        tabsFrame.add(hillClimbFrame, text='Hill Climb')
        tabsFrame.add(annealingFrame, text='Annealing')

        tabsFrame.grid(row=0, column=0, padx=self.defPad, pady=self.defPad, sticky=tk.NW + tk.NE)

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
                "sigma": tk.DoubleVar()
        }
        self.hillClimbOptions["pointCloud"].set(60)
        self.hillClimbOptions["sigma"].set(0.05)

        pointCloudSizeLabel = ttk.Label(frame, text="Point cloud size")
        pointCloudSizeLabel.grid(row=0, column=0, padx=self.defPad, pady=self.defPad, sticky=tk.E)

        pointCloudSizeEntry = ttk.Entry(frame, textvariable=self.hillClimbOptions["pointCloud"])
        pointCloudSizeEntry.grid(row=0, column=1, padx=self.defPad, pady=self.defPad, sticky=tk.W+tk.E)

        sigmaLabel = ttk.Label(frame, text="Sigma")
        sigmaLabel.grid(row=1, column=0, padx=self.defPad, pady=self.defPad, sticky=tk.E)

        sigmaSlider = ttk.Scale(frame, from_ =0.0, to = 1.0, variable=self.hillClimbOptions["sigma"])
        sigmaSlider.grid(row=1, column=1, padx=self.defPad, pady=self.defPad, sticky=tk.W+tk.E)

        sigmaEntry = ttk.Entry(frame, textvariable=self.hillClimbOptions["sigma"])
        sigmaEntry.grid(row=2, column=1, padx=self.defPad, pady=self.defPad, sticky=tk.W+tk.E)

        return frame

    def createAnnealingFrame(self, master):
        frame = ttk.Frame(master, width=self.defaultNotebookFrameWidth, height=self.defaultNotebookFrameHeight)

        self.annealingOptions = {
                "pointCloud": tk.IntVar()
        }
        self.annealingOptions["pointCloud"].set(60)

        return frame

    def run(self):
        print('beep boop')


if __name__ == '__main__':
    root = tk.Tk()
    root.title("BIA GUI")
    app = Application(master=root)
    root.resizable(False, False)
    app.mainloop()
