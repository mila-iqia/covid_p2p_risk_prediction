import argparse
from raven.core import RavenJob


COVID19SIM_RUN = ""


def parse_args():
    parsey = argparse.ArgumentParser()
    parsey.add_argument("-e", "--experiment-directory", type=str, required=True)
    return parsey.parse_args()


def get_script_args(args):
    script_args = {
        "INTERVENTION_DAY": 3,
        "tune": True,
        "APP_UPTAKE": 0.85,
        "tracing_method": "transformer",
        "init_percent_sick": 0.002,
        "seed": 200,
        "outdir": None
    }
    pass


def evaluate(args):

    raven_job = RavenJob().set_script_path(COVID19SIM_RUN).set_script_args()
    pass