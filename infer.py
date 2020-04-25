import os
from speedrun import BaseExperiment

from loader import ContactPreprocessor, InvalidSetSize
from models import ContactTracingTransformer
import torch

import pickle
import threading
import typing
import zmq


class InferenceEngine(BaseExperiment):
    def __init__(self, experiment_directory):
        super(InferenceEngine, self).__init__(experiment_directory=experiment_directory)
        self.record_args().read_config_file()
        self._build()

    def _build(self):
        self.preprocessor = ContactPreprocessor(
            relative_days=self.get("data/loader_kwargs/relative_days", True)
        )
        self.model: torch.nn.Module = ContactTracingTransformer(
            **self.get("model/kwargs", {})
        )
        self.load()

    def load(self):
        path = os.path.join(self.checkpoint_directory, "best.ckpt")
        assert os.path.exists(path)
        state = torch.load(path)
        self.model.load_state_dict(state["model"])
        self.model.eval()
        return self

    def infer(self, human_day_info):
        with torch.no_grad():
            model_input = self.preprocessor.preprocess(human_day_info, as_batch=True)
            model_output = self.model(model_input.to_dict())
            contagion_proba = (
                model_output["encounter_variables"].sigmoid().numpy()[0, :, 0]
            )
            infectiousness = model_output["latent_variable"].numpy()[0, :, 0]
        return dict(contagion_proba=contagion_proba, infectiousness=infectiousness)


class InferenceServer(threading.Thread):

    def __init__(
            self,
            experiment_directory: typing.AnyStr,
            port: int = 6688,
            poll_delay_ms: int = 1000,
    ):
        threading.Thread.__init__(self)
        self.experiment_directory = experiment_directory
        self.port = port
        self.poll_delay_ms = poll_delay_ms
        self.stop_flag = threading.Event()

    def run(self):
        engine = InferenceEngine(self.experiment_directory)
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.identity = f"inference:{self.port}".encode("ascii")
        socket.bind(f"tcp://*:{self.port}")
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)
        while not self.stop_flag.is_set():
            evts = dict(poller.poll(self.poll_delay_ms))
            if socket in evts and evts[socket] == zmq.POLLIN:
                hdi = pickle.loads(socket.recv())
                output = None
                try:
                    output = engine.infer(hdi)
                except InvalidSetSize:
                    pass  # return None for invalid samples
                socket.send(pickle.dumps(output))
        socket.close()

    def stop(self):
        self.stop_flag.set()


class InferenceClient:
    def __init__(
            self,
            target_port: typing.Union[int, typing.List[int]],
            target_addr: typing.AnyStr = "localhost",
    ):
        if isinstance(target_port, int):
            self.target_ports = [target_port]
        else:
            self.target_ports = target_port
        self.target_addr = target_addr
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        for port in self.target_ports:
            self.socket.connect(f"tcp://{target_addr}:{port}")

    def infer(self, sample):
        self.socket.send(pickle.dumps(sample))
        return pickle.loads(self.socket.recv())


if __name__ == "__main__":
    from loader import ContactDataset

    dataset = ContactDataset(
        path="/Users/nrahaman/Python/ctt/data/sim_people-1000_days-60_init-0"
    )
    hdi = dataset.read(0, 0)

    engine = InferenceEngine("/Users/nrahaman/Python/ctt/exp/DEBUG-0")
    output = engine.infer(hdi)
    pass
