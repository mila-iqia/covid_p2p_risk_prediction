"""Entrypoint that can be used to start many inference workers on a single node."""

import argparse
import functools
import os
import signal
import sys
import time

import server_utils

default_port = 6688
default_workers = 1
default_threads = 8
default_model_exp_path = os.path.abspath("exp/DEBUG-0")
default_mp_backend = "loky"


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
    mp_backend_doc = f"Name of the joblib backend to use. Default is {default_mp_backend}."
    argparser.add_argument("--mp-backend", default=None, type=int, help=mp_backend_doc)
    mp_threads_doc = f"Number of threads to spawn in each worker. Will use {default_threads} by default."
    argparser.add_argument("--mp-threads", default=None, type=int, help=mp_threads_doc)
    args = argparser.parse_args(args)
    return args.port, args.exp_path, args.workers, args.verbose, args.mp_backend, args.mp_threads


def validate_args(port, exp_path, workers, verbose, mp_backend, mp_threads):
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
    if mp_threads is None:
        mp_threads = default_threads
    assert mp_threads > 0, f"invalid thread count: {mp_threads}"
    return port, exp_path, workers, verbose, mp_backend, mp_threads


def interrupt_handler(signal, frame, manager):
    print("Received SIGINT; shutting down inference engine(s) gracefully...")
    manager.stop()
    manager.join()
    print("All done.")
    sys.exit(0)


def main(args=None):
    port, exp_path, workers, verbose, mp_backend, mp_threads = \
        validate_args(*parse_args(args))
    manager = server_utils.InferenceServerManager(
        model_exp_path=exp_path,
        workers=workers,
        mp_backend=mp_backend,
        mp_threads=mp_threads,
        port=port,
        verbose=verbose,
    )
    manager.start()
    handler = functools.partial(interrupt_handler, manager=manager)
    signal.signal(signal.SIGINT, handler)
    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()
