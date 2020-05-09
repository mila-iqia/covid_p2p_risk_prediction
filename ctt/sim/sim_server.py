from speedrun import BaseExperiment
from covid19sim.run import run_simu


class SimServer(BaseExperiment):
    def __init__(self):
        super(SimServer, self).__init__()
        self.auto_setup()

    def launch(self):
        pass

    @staticmethod
    def job():
        # This method should (1) start the server with the weights
        # it has available, (2) configure and call the simulator.
        pass
