"""Entrypoint that can be used to start many inference servers on a single node."""

import argparse
import functools
import os
import signal
import sys
import time

import server_utils

default_port = 6688
default_model_exp_path = os.path.abspath("exp/DEBUG-0")


def parse_args(args=None):
    argparser = argparse.ArgumentParser(
        description="COVID19-P2P-Transformer Inference Server Spawner",
    )
    ports_doc = f"Input ports to accept connections on; will use {default_port} if not provided. " \
                f"A single port or a range of ports (given as \"XXXX-YYYY\") can be used."
    argparser.add_argument("-p", "--ports", default=None, type=str, help=ports_doc)
    exp_path_doc = f"Path to the experiment directory that should be used to instantiate the " \
                   f"inference engine(s). Will use '{default_model_exp_path}' if not provided. " \
                   f"See `infer.py` for more information."
    argparser.add_argument("-e", "--exp-path", default=None, type=str, help=exp_path_doc)
    args = argparser.parse_args(args)
    return args.ports, args.exp_path


def validate_args(ports, exp_path):
    if ports is None:
        ports = [default_port]
    else:
        assert ports.count('-') <= 1, "port range should contain a single dash"
        if '-' in ports:
            ports = [int(port) for port in ports.split('-')]
        else:
            assert ports.isdigit(), f"unexpected port number format ({ports})"
            ports = [int(ports)]
    if exp_path is None:
        exp_path = default_model_exp_path
    assert os.path.isdir(exp_path), f"invalid experiment directory path: {exp_path}"
    return ports, exp_path


def interrupt_handler(signal, frame, servers):
    print("Received SIGINT; shutting down inference engine(s) gracefully...")
    for s in servers:
        s.stop()
        s.join()
    print("All done.")
    sys.exit(0)


def main(args=None):
    ports, exp_path = validate_args(*parse_args(args))
    print(f"Initializing {len(ports)} inference engine(s) from directory: {exp_path}")
    servers = [server_utils.InferenceServer(exp_path, port) for port in ports]
    _ = [s.start() for s in servers]
    handler = functools.partial(interrupt_handler, servers=servers)
    signal.signal(signal.SIGINT, handler)
    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()
