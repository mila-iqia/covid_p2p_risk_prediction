import sys
import argparse
import subprocess
import persistqueue
from contextlib import contextmanager

from concurrent.futures import ProcessPoolExecutor
from typing import Union

import ctt.sim.utils as su


# ----------- Simulation and Serving -----------


@contextmanager
def launch_inference_server(experiment_directory: str, weight_path: str):
    server_process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "covid19sim.server_bootstrap",
            "-e",
            experiment_directory,
            "-w",
            weight_path,
        ]
    )
    yield server_process
    # Now kill it
    server_process.terminate()


def run_simulation(simulation_kwargs: dict) -> dict:
    """
    This function wraps all covid19sim business and returns a dict of metrics.
    Assumes that the inference server is running.
    """
    from covid19sim.run import run_simu
    from ctt.sim.metrics import SimulationMetrics

    _, _, city = run_simu(return_city=True, **simulation_kwargs["sim"])
    metrics = SimulationMetrics(**simulation_kwargs["metrics"])(city=city)
    return metrics


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
        default=300,
        help="How long to wait before giving up on the out-going queue.",
    )
    parsey.add_argument(
        "-w",
        "--max-num-workers",
        type=int,
        default=4,
        help="Max number of workers to spin up. Defaults to the number of jobs.",
    )
    args = parsey.parse_args()
    return args


def get_job(queue_directory: str, timeout=300) -> Union[dict, None]:
    """Reads in a job from the outgoing-queue in `queue_directory`."""
    queue = persistqueue.SQLiteQueue(
        path=queue_directory, name=su.get_outgoing_queue_name(), auto_commit=True
    )
    try:
        # Wait to read something for 5 minutes.
        return queue.get(timeout=timeout)
    except persistqueue.Empty:
        return None


def return_results(queue_directory: str, job: dict, results: dict):
    """
    Returns the result of the simulation to the incoming-queue in `queue_directory`.
    """
    payload = {**job, "results": results}
    queue = persistqueue.SQLiteQueue(
        path=queue_directory, name=su.get_incoming_queue_name(), auto_commit=True
    )
    queue.put(payload)


def launch(args: argparse.Namespace):
    """
    This is the main function that is called inside the slurm/HTCondor job.
    """

    # Read in a job from the queue
    queue_directory = su.get_queue_directory(args.experiment_directory)
    job = get_job(queue_directory=queue_directory, timeout=args.read_timeout)
    if job is None:
        # No job found in queue after waiting till time-out, I guess there's
        # nothing to do.
        return
    # Launch the inference server and run the job
    with launch_inference_server(
        experiment_directory=job["experiment_directory"], weight_path=job["weight_path"]
    ):
        # Run the sim
        if isinstance(job["simulation_kwargs"], list):
            max_num_workers = args.max_num_workers or len(job["simulation_kwargs"])
            with ProcessPoolExecutor(max_workers=max_num_workers) as executor:
                results = list(executor.map(run_simulation, job["simulation_kwargs"]))
        else:
            # Run single process
            results = run_simulation(job["simulation_kwargs"])
    # Return the results which will be read in by the training loop
    return_results(queue_directory, job, results)


if __name__ == "__main__":
    launch(parse_args())
