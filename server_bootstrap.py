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
default_model_exp_path = os.path.abspath("exp/DEBUG-0")


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


def interrupt_handler(signal, frame, manager):
    print("Received SIGINT; shutting down inference engine(s) gracefully...")
    manager.stop()
    manager.join()
    print("All done.")
    sys.exit(0)


def main(args=None):
    port, exp_path, workers, verbose = validate_args(*parse_args(args))
    manager = server_utils.InferenceServerManager(
        model_exp_path=exp_path,
        workers=workers,
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
