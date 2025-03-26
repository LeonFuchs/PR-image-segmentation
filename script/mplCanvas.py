import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self,parent=None,width=6,height=3):
        fig = Figure(figsize=(width,height))
        self.axes = fig.add_subplot(111)
        super().__init__(fig)