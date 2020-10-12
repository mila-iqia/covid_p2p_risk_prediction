import os


def get_sim_outdir(job):
    return os.path.join(job["experiment_directory"], "Logs", "SimRuns", job["iden"])


def get_queue_directory(experiment_directory: str) -> str:
    return os.path.join(experiment_directory, "Logs", "SimQ")


def get_evaluation_checkpoint_directory(experiment_directory: str) -> str:
    return os.path.join(experiment_directory, "Weights", "Evaluation")


def get_outgoing_queue_name() -> str:
    return "simq_out"


def get_incoming_queue_name() -> str:
    return "simq_in"


def patch_persistqueue_pickle_to_dill():
    import persistqueue.serializers.pickle
    import dill
    persistqueue.serializers.pickle.pickle = dill


class SimulationError(Exception):
    pass