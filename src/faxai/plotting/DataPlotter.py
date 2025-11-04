
from abc import ABC, abstractmethod

class DataPlotter(ABC):

    @abstractmethod
    def matplotlib_plot(self, ax: plt.axes) -> None:
        pass

    @abstractmethod
    def plotly_plot(self, ax: go.Figure) -> None:
        pass
