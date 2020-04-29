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

default_poll_delay_ms = 500


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


class InferenceWorker(threading.Thread):
    """Spawns a single inference worker instance.

    These workers are managed by the InferenceBroker class. They
    communicate with the broker using a backend connection.
    """

    def __init__(
            self,
            experiment_directory: typing.AnyStr,
            identifier: typing.Any,
            mp_backend: typing.AnyStr,
            mp_threads: int,
            context: typing.Optional[zmq.Context] = None,
    ):
        threading.Thread.__init__(self)
        self.experiment_directory = experiment_directory
        self.identifier = identifier
        self.stop_flag = threading.Event()
        self.packet_counter = AtomicCounter(init=0)
        self.time_counter = AtomicCounter(init=0.)
        self.mp_backend = mp_backend
        self.mp_threads = mp_threads
        if context is None:
            context = zmq.Context()
        self.context = context
        self.init_time = None

    def run(self):
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
        self.init_time = time.time()
        while not self.stop_flag.is_set():
            evts = dict(poller.poll(default_poll_delay_ms))
            if socket in evts and evts[socket] == zmq.POLLIN:
                proc_start_time = time.time()
                address, empty, buffer = socket.recv_multipart()
                hdi = pickle.loads(buffer)
                response = self.process_sample(
                    hdi, engine, self.mp_backend, self.mp_threads)
                response = pickle.dumps(response)
                socket.send_multipart([address, b"", response])
                self.time_counter.increment(time.time() - proc_start_time)
                self.packet_counter.increment()
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

    def get_processing_uptime(self):
        """Returns the fraction of total uptime that the server spends processing requests."""
        tot_process_time, tot_time = self.time_counter.count, time.time() - self.init_time
        return tot_process_time / tot_time

    def stop(self):
        """Stops the infinite data reception loop, allowing a clean shutdown."""
        self.stop_flag.set()

    @staticmethod
    def process_sample(sample, engine, mp_backend=None, mp_threads=0):
        if isinstance(sample, list):
            if mp_threads > 0:
                import joblib
                with joblib.Parallel(
                        n_jobs=mp_threads,
                        backend=mp_backend,
                        batch_size="auto",
                        prefer="threads") as parallel:
                    results = parallel((joblib.delayed(proc_human)(human) for human in sample))
                return [(r['name'], r['risk'], r['clusters']) for r in results]
            else:
                return [InferenceWorker.process_sample(human, engine, mp_backend, mp_threads) for human in sample]
        else:
            assert isinstance(sample, dict), "unexpected input data format"
            results = proc_human(sample)
            if results is not None:
                return (results['name'], results['risk'], results['clusters'])
            return None


class InferenceBroker(threading.Thread):
    """Manages inference workers through a backend connection for load balancing."""

    def __init__(
            self,
            model_exp_path: typing.AnyStr,
            workers: int,
            mp_backend: typing.AnyStr,
            mp_threads: int,
            port: int,
            verbose: bool = False,
            verbose_print_delay: float = 5.,
            context: typing.Optional[zmq.Context] = None,
    ):
        threading.Thread.__init__(self)
        if context is None:
            context = zmq.Context()
        self.context = context
        self.workers = workers
        self.mp_backend = mp_backend
        self.mp_threads = mp_threads
        self.port = port
        self.model_exp_path = model_exp_path
        self.stop_flag = threading.Event()
        self.verbose = verbose
        self.verbose_print_delay = verbose_print_delay

    def run(self):
        print(f"Initializing {self.workers} worker(s) from directory: {self.model_exp_path}")
        frontend = self.context.socket(zmq.ROUTER)
        frontend.bind(f"tcp://*:{self.port}")
        backend = self.context.socket(zmq.ROUTER)
        backend.bind("ipc://backend.ipc")
        worker_map = {}
        for worker_idx in range(self.workers):
            worker_id = f"worker:{worker_idx}"
            worker = InferenceWorker(
                self.model_exp_path,
                worker_id,
                self.mp_backend,
                self.mp_threads,
                context=self.context
            )
            worker_map[worker_id] = worker
            worker.start()
        available_worker_ids = []
        worker_poller = zmq.Poller()
        worker_poller.register(backend, zmq.POLLIN)
        worker_poller.register(frontend, zmq.POLLIN)
        last_update_timestamp = time.time()
        while not self.stop_flag.is_set():
            evts = dict(worker_poller.poll(default_poll_delay_ms))
            if backend in evts and evts[backend] == zmq.POLLIN:
                request = backend.recv_multipart()
                worker_id, empty, client = request[:3]
                available_worker_ids.append(worker_id)
                if client != b"READY" and len(request) > 3:
                    empty, reply = request[3:]
                    frontend.send_multipart([client, b"", reply])
            if available_worker_ids and frontend in evts and evts[frontend] == zmq.POLLIN:
                client, empty, request = frontend.recv_multipart()
                worker_id = available_worker_ids.pop(0)
                backend.send_multipart([worker_id, b"", client, b"", request])
            if self.verbose and time.time() - last_update_timestamp > self.verbose_print_delay:
                print(f" {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} stats:")
                for worker_id, worker in worker_map.items():
                    packets = worker.get_processed_count()
                    delay = worker.get_averge_processing_delay()
                    uptime = worker.get_processing_uptime()
                    print(f"  {worker_id}:  packets={packets}  avg_delay={delay:.6f}sec  proc_time_ratio={uptime:.1%}")
                last_update_timestamp = time.time()
        for w in worker_map.values():
            w.stop()
            w.join()

    def stop(self):
        """Stops the infinite data reception loop, allowing a clean shutdown."""
        self.stop_flag.set()


class InferenceClient:
    """Creates a client through which data samples can be sent for inference.

    This object will automatically be able to pick a proper remote inference
    engine. This object should be fairly lightweight and low-cost, so creating
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
    if all([p in params for p in expected_processed_packet_param_names]):
        return params, None  # probably fetching data from data loader; skip stuff below

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

    inference_result = None
    if params['risk_model'] == "transformer":
        try:
            inference_result = engine.infer(daily_output)
        except InvalidSetSize:
            pass  # return None for invalid samples
    if inference_result is not None:
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # ... TODO: apply the inference results to the human's risk before returning it
        #           (it will depend on the output format used by Nasim)
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        pass

    return human
