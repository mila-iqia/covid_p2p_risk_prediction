"""Contains utility classes for remote inference inside the simulation."""

import datetime
import numpy as np
import os
import pickle
import threading
import time
import typing
import zmq

from infer import InferenceEngine
from loader import InvalidSetSize
from risk_models import RiskModelTristan

expected_raw_packet_param_names = [
    "start", "current_day", "all_possible_symptoms", "human",
    "COLLECT_LOGS", "log_path", "risk_model"
]

expected_processed_packet_param_names = [
    "current_day", "observed", "unobserved"
]


class AtomicCounter(object):
    """Implements an atomic & thread-safe counter."""

    def __init__(self, init=0):
        self._count = init
        self._lock = threading.Lock()

    def increment(self, delta=1):
        with self._lock:
            self._count += delta
            return self._count

    @property
    def count(self):
        return self._count


class InferenceServer(threading.Thread):
    """Spawns a single inference server instance for a specific port.

    Multiple inference servers can be started simultaneously on the same
    machine using different ports. The inference client will automatically
    be able to pick a proper remote inference engine if it is provided all
    the server addresses/ports simultaneously. This load balancing is
    implemented by design in zmq.

    Once created, the server instance will be able to accept connections
    and return inference results until `server.stop()` is called. The server
    itself runs in a separate thread that can be joined to insure proper
    cleanup by the bootstrapping script.
    """

    def __init__(
            self,
            experiment_directory: typing.AnyStr,
            identifier: typing.Any,
            poll_delay_ms: int = 1000,
            context: typing.Optional[zmq.Context] = None,
    ):
        threading.Thread.__init__(self)
        self.experiment_directory = experiment_directory
        self.identifier = identifier
        self.poll_delay_ms = poll_delay_ms
        self.stop_flag = threading.Event()
        self.packet_counter = AtomicCounter(init=0)
        self.time_counter = AtomicCounter(init=0.)
        if context is None:
            context = zmq.Context()
        self.context = context

    def run(self):
        """Will be automatically called by the base class after construction."""
        # import frozen modules with classes required for unpickling
        import frozen.clusters
        import frozen.utils
        engine = InferenceEngine(self.experiment_directory)
        socket = self.context.socket(zmq.REQ)
        socket.identity = str(self.identifier).encode("ascii")
        socket.connect("ipc://backend.ipc")
        socket.send(b"READY")  # tell broker we're ready
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)
        while not self.stop_flag.is_set():
            evts = dict(poller.poll(self.poll_delay_ms))
            if socket in evts and evts[socket] == zmq.POLLIN:
                proc_start_time = time.time()
                address, empty, buffer = socket.recv_multipart()
                hdi = pickle.loads(buffer)
                output, human = None, None
                try:
                    if len(hdi) != len(expected_processed_packet_param_names) and hdi['risk_model'] == "naive":
                        output, human = proc_human(hdi)
                    elif hdi['risk_model'] == "transformer":
                        output, human = proc_human(hdi)
                        output = engine.infer(hdi)
                    else:
                        raise NotImplementedError
                except InvalidSetSize:
                    pass  # return None for invalid samples
                self.packet_counter.increment()
                self.time_counter.increment(time.time() - proc_start_time)
                if human is not None:
                    response = pickle.dumps((human['name'], human['risk'], human['clusters']))
                else:
                    response = pickle.dumps(None)
                socket.send_multipart([address, b"", response])
        socket.close()

    def get_processed_count(self):
        """Returns the total number of processed requests by this inference server."""
        return self.packet_counter.count

    def get_averge_processing_delay(self):
        """Returns the average sample processing time between reception & response (in seconds)."""
        tot_delay, tot_packet_count = self.time_counter.count, self.packet_counter.count
        if not tot_packet_count:
            return float("nan")
        return tot_delay / self.packet_counter.count

    def stop(self):
        """Stops the infinite data reception loop, allowing a clean shutdown."""
        self.stop_flag.set()


class InferenceClient:
    """Creates a client through which data samples can be sent for inference.

    This object will automatically be able to pick a proper remote inference
    engine if it is provided all the started server addresses/ports. This load
    balancing is implemented by design in zmq.

    This object should be fairly lightweight and low-cost, so creating/deleting
    it once per day, per human *should* not create a significant overhead.
    """
    def __init__(
            self,
            target_port: typing.Union[int, typing.List[int]],
            target_addr: typing.Union[str, typing.List[str]] = "localhost",
            context: typing.Optional[zmq.Context] = None,
    ):
        self.target_ports = [target_port] if isinstance(target_port, int) else target_port
        self.target_addrs = [target_addr] if isinstance(target_addr, str) else target_addr
        if len(self.target_ports) != len(self.target_addrs):
            assert len(self.target_addrs) == 1 and len(self.target_ports) > 1, \
                "must either match all ports to one address or provide full port/addr combos"
            self.target_addrs = self.target_addrs * len(self.target_ports)
        if context is None:
            context = zmq.Context()
        self.context = context
        self.socket = self.context.socket(zmq.REQ)
        for addr, port in zip(self.target_addrs, self.target_ports):
            self.socket.connect(f"tcp://{addr}:{port}")

    def infer(self, sample):
        """Forwards a data sample for the inference engine using pickle."""
        self.socket.send(pickle.dumps(sample))
        return pickle.loads(self.socket.recv())


def proc_human(params):
    """(Pre-)Processes the received simulator data for a single human."""
    import frozen.helper
    assert isinstance(params, dict) and \
        all([p in params for p in expected_raw_packet_param_names]), \
        "unexpected/broken proc_human input format between simulator and inference service"

    human = params["human"]
    human["clusters"].add_messages(human["messages"], params["current_day"], human["rng"])
    human["messages"] = []
    human["clusters"].update_records(human["update_messages"], human)
    human["update_messages"] = []
    human["clusters"].purge(params["current_day"])

    todays_date = params['start'] + datetime.timedelta(params['current_day'])

    human['risk'] = RiskModelTristan.update_risk_daily(human, todays_date)

    # update risk based on that day's messages
    if human['messages']:
        human['risk'] = RiskModelTristan.update_risk_encounters(human)

    todays_date = params["start"] + datetime.timedelta(days=params["current_day"])
    is_exposed, exposure_day = frozen.helper.exposure_array(human["infection_timestamp"], todays_date)
    is_recovered, recovery_day = frozen.helper.recovered_array(human["recovered_timestamp"], todays_date)
    candidate_encounters, exposure_encounter = frozen.helper.candidate_exposures(human, todays_date)
    reported_symptoms = frozen.helper.symptoms_to_np(human["all_reported_symptoms"], params["all_possible_symptoms"])
    true_symptoms = frozen.helper.symptoms_to_np(human["all_symptoms"], params["all_possible_symptoms"])
    daily_output = {
        "current_day": params["current_day"],
        "observed": {
            "reported_symptoms": reported_symptoms,
            "candidate_encounters": candidate_encounters,
            "test_results": frozen.helper.get_test_result_array(human["test_time"], todays_date),
            "preexisting_conditions": frozen.helper.conditions_to_np(human["obs_preexisting_conditions"]),
            "age": frozen.helper.encode_age(human["obs_age"]),
            "sex": frozen.helper.encode_sex(human["obs_sex"])
        },
        "unobserved": {
            "true_symptoms": true_symptoms,
            "is_exposed": is_exposed,
            "exposure_day": exposure_day,
            "is_recovered": is_recovered,
            "recovery_day": recovery_day,
            "infectiousness": np.array(human["infectiousnesses"]),
            "true_preexisting_conditions": frozen.helper.conditions_to_np(human["preexisting_conditions"]),
            "true_age": frozen.helper.encode_age(human["age"]),
            "true_sex": frozen.helper.encode_sex(human["sex"])
        }
    }

    if params["COLLECT_LOGS"]:
        os.makedirs(params["log_path"], exist_ok=True)
        with open(os.path.join(params["log_path"], f"daily_human.pkl"), 'wb') as fd:
            pickle.dump(daily_output, fd)

    return daily_output, human
