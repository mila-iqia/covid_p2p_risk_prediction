import os


def get_queue_directory(experiment_directory: str) -> str:
    return os.path.join(experiment_directory, "Logs", "SimQ")


def get_evaluation_checkpoint_directory(experiment_directory: str) -> str:
    return os.path.join(experiment_directory, "Weights", "Evaluation")


def get_outgoing_queue_name() -> str:
    return "simq_out"


def get_incoming_queue_name() -> str:
    return "simq_in"
