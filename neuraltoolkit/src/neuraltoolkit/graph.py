import numpy as np
import matplotlib.pyplot as plt


class Graph:
    def __init__(self, x_label, y_label, title):
        self.x_label = x_label
        self.y_label = y_label
        self.title = title
        self.lines = []
        self.plots = {}

        self.width = 1
        self.height = 1
        
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel(self.x_label)
        self.ax.set_ylabel(self.y_label)
        self.ax.set_title(self.title)
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        self.ax.grid(True)
        self.ax.autoscale(enable=True, axis='both', tight=None) 
        #plt.ion()
        #plt.show()

    def close(self, event=None):
        plt.close(self.fig)

    def add_plot(self, name:str, color="blue"):
        self.plots[name] = {
            "plot": self.ax.plot([], [], color, label=name),
            "x": [],
            "y": []
        }

    def add_plot(self, name:str, x_data:list, y_data:list, color="blue"):
        self.plots[name] = {
        "plot": self.ax.plot(x_data, y_data, color, label=name),
        "x": [],
        "y": []
    }

    def append(self, name:str, x, y):
        self.plots[name]["x"].append(x)
        self.plots[name]["y"].append(y)

        self.plots[name]["plot"][0].set_data(
            self.plots[name]["x"],
            self.plots[name]["y"]
        )

    def set(self, name:str, x_data, y_data):
        self.plots[name]["plot"][0].set_data(x_data, y_data)

    def update(self):
        self.fig.canvas.mpl_connect('close_event', self.close)
        self.ax.relim()
        self.ax.autoscale(enable=True, axis='both', tight=None) 
        self.ax.legend()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def show(self):
        self.fig.canvas.mpl_connect('close_event', self.close)
        self.ax.relim()
        self.ax.autoscale(enable=True, axis='both', tight=None) 
        self.ax.legend()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.show()
