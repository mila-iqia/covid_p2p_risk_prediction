"""Entrypoint that can be used to start many inference workers on a single node."""

import argparse
import datetime
import functools
import os
import signal
import sys
import time
import zmq

import server_utils

default_port = 6688
default_workers = 1
default_model_exp_path = os.path.abspath("exp/DEBUG-0")
default_poll_delay_ms = 10000
default_verbose_print_delay_sec = 30

def parse_args(args=None):
    argparser = argparse.ArgumentParser(
        description="COVID19-P2P-Transformer Inference Server Spawner",
    )
    port_doc = f"Input port to accept connections on; will use {default_port} if not provided. "
    argparser.add_argument("-p", "--port", default=None, type=str, help=port_doc)
    exp_path_doc = f"Path to the experiment directory that should be used to instantiate the " \
                   f"inference engine(s). Will use '{default_model_exp_path}' if not provided. " \
                   f"See `infer.py` for more information."
    argparser.add_argument("-e", "--exp-path", default=None, type=str, help=exp_path_doc)
    workers_doc = f"Number of inference workers to spawn. Will use {default_workers} by default."
    argparser.add_argument("-w", "--workers", default=None, type=int, help=workers_doc)
    verbosity_doc = "Toggles program verbosity on/off. Default is OFF (0). Variable expects 0 or 1."
    argparser.add_argument("-v", "--verbose", default=0, type=int, help=verbosity_doc)
    args = argparser.parse_args(args)
    return args.port, args.exp_path, args.workers, args.verbose


def validate_args(port, exp_path, workers, verbose):
    if port is None:
        port = default_port
    else:
        assert port.isdigit(), f"unexpected port number format ({port})"
        port = int(port)
    if exp_path is None:
        exp_path = default_model_exp_path
    assert os.path.isdir(exp_path), f"invalid experiment directory path: {exp_path}"
    if workers is None:
        workers = default_workers
    assert workers > 0, f"invalid worker count: {workers}"
    return port, exp_path, workers, verbose


def interrupt_handler(signal, frame, workers):
    print("Received SIGINT; shutting down inference engine(s) gracefully...")
    for w in workers.values():
        w.stop()
        w.join()
    print("All done.")
    sys.exit(0)


def main(args=None):
    port, exp_path, workers, verbose = validate_args(*parse_args(args))
    print(f"Initializing {workers} inference engine(s) from directory: {exp_path}")
    context = zmq.Context()
    frontend = context.socket(zmq.ROUTER)
    frontend.bind(f"tcp://*:{port}")
    backend = context.socket(zmq.ROUTER)
    backend.bind("ipc://backend.ipc")
    worker_map = {}
    for worker_idx in range(workers):
        worker_id = f"worker:{worker_idx}"
        worker = server_utils.InferenceServer(exp_path, worker_id, context=context)
        worker_map[worker_id] = worker
        worker.start()
    available_worker_ids = []
    handler = functools.partial(interrupt_handler, workers=worker_map)
    signal.signal(signal.SIGINT, handler)
    worker_poller = zmq.Poller()
    worker_poller.register(backend, zmq.POLLIN)
    worker_poller.register(frontend, zmq.POLLIN)
    last_update_timestamp = time.time()
    while True:
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
        if verbose and time.time() - last_update_timestamp > default_verbose_print_delay_sec:
            print(f" {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} stats:")
            for worker_id, worker in worker_map.items():
                packets = worker.get_processed_count()
                delay = worker.get_averge_processing_delay()
                print(f"  {worker_id}:  packets={packets}  avg_delay={delay:.6f}sec")
            last_update_timestamp = time.time()


if __name__ == "__main__":
    main()
