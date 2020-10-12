import sys
import os
import glob
import argparse
import pickle
import subprocess
import persistqueue

from contextlib import contextmanager
import time
from typing import Union

import ctt.sim.utils as su


# ----------- Simulation and Serving -----------


def run_simulation(job: dict):
    """
    This function wraps all covid19sim business and returns a dict of metrics.
    Assumes that the inference server is running.
    """
    from ctt.inference.infer import InferenceEngine

    # Convert simulation args to command-line args consumable by hydra.
    hydra_args = [
        f"{arg_name}={arg_val}"
        for arg_name, arg_val in job["simulation_kwargs"].items()
    ]
    hydra_args = [
        "tune=True",
        "USE_INFERENCE_SERVER=False",
        "tracing_method=transformer",
        f"TRANSFORMER_EXP_PATH={job['experiment_directory']}"
        f"{InferenceEngine.SPLIT_KEY}{job['weight_path']}",
        f"outdir={su.get_sim_outdir(job)}",
    ] + hydra_args
    process = subprocess.Popen(
        [sys.executable, "-m", "covid19sim.run"] + hydra_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return process


def parse_results(job):
    sim_outdir = su.get_sim_outdir(job)
    all_pickles = glob.glob(os.path.join(sim_outdir, "*/*.pkl"))
    data_pickle = [
        p for p in all_pickles if os.path.basename(p).startswith("tracker_data")
    ]
    assert len(data_pickle) > 0, "No pickles found!"
    with open(all_pickles[0], "rb") as f:
        result = pickle.load(f)
    return result


# ----------- Interface and IO -----------


def parse_args() -> argparse.Namespace:
    parsey = argparse.ArgumentParser()
    parsey.add_argument(
        "-e",
        "--experiment-directory",
        type=str,
        required=True,
        help="Experiment directory used to build the model.",
    )
    parsey.add_argument(
        "-t",
        "--read-timeout",
        type=int,
        default=5,
        help="How long to wait before giving up on the out-going queue.",
    )
    parsey.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Logs to stderror if set to activated.",
    )
    parsey.add_argument(
        "-s",
        "--max-num-sim-workers",
        type=int,
        default=4,
        help=(
            "Max number of workers to spin up for the spin. "
            "Defaults to the number of simulation parameters sent."
        ),
    )
    args = parsey.parse_args()
    return args


@contextmanager
def get_job(queue_directory: str, timeout=300):
    """Reads in a job from the outgoing-queue in `queue_directory`."""
    queue = persistqueue.SQLiteAckQueue(
        path=queue_directory, name=su.get_outgoing_queue_name(), auto_commit=True
    )
    try:
        # Wait to read something for 5 minutes.
        job = queue.get(timeout=timeout)
    except persistqueue.Empty:
        job = None
    do_ack = True
    try:
        yield job
    except Exception:
        do_ack = False
        raise
    finally:
        if job is None:
            # Job was not read, nothing to do here
            pass
        elif do_ack:
            # Job was read, ack
            queue.ack(job)
        else:
            # Job failed, don't ack
            queue.nack(job)


def return_results(queue_directory: str, job: dict, results: dict):
    """
    Returns the result of the simulation to the incoming-queue in `queue_directory`.
    """
    payload = {**job, "results": results}
    queue = persistqueue.SQLiteQueue(
        path=queue_directory, name=su.get_incoming_queue_name(), auto_commit=True
    )
    queue.put(payload)


def run(args: argparse.Namespace):
    # Read in a job from the queue
    queue_directory = su.get_queue_directory(args.experiment_directory)
    if args.verbose:
        print(f"Reading from directory: {queue_directory}")
    with get_job(queue_directory=queue_directory, timeout=args.read_timeout) as job:
        if job is None:
            # No job found in queue after waiting till time-out, I guess there's
            # nothing to do.
            return
        if args.verbose:
            print(f"Running job:\n{job}")
        # Run the sim and wait for results to come out
        process = run_simulation(job)
        while True:
            time.sleep(10)
            if process.poll() is not None:
                break
            if args.verbose:
                for line in iter(process.stdout.readline, b""):
                    print(f">>> {line.rstrip().decode('utf-8')}")
        process.wait()
        if args.verbose:
            print(f"Job finished with return code: {process.returncode}")
        if process.returncode != 0:
            stdout, stderr = process.communicate()
            raise su.SimulationError(
                f"Simulation failed. The stderr was:\n{stderr.decode('utf-8')}"
            )
        # Read in the results from the tracker data
        if args.verbose:
            print(f"Parsing results...")
        results = parse_results(job)
        # Return the results which will be read in by the training loop
        if args.verbose:
            print(f"Writing results to incoming queue...")
        return_results(queue_directory, job, results)
        if args.verbose:
            print(f"Done.")


if __name__ == "__main__":
    run(parse_args())
    pass
