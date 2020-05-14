import os
import zipfile
import pickle
from datetime import datetime

from addict import Dict
import networkx as nx
import numpy as np


class SimulationMetrics(object):
    def __init__(
        self,
        simulator_log_directory=None,
        simulation_start=datetime(2020, 2, 28),
        simulation_stop=datetime(2020, 3, 12),
    ):
        self.simulator_log_directory = simulator_log_directory
        self.simulation_start = simulation_start
        self.simulation_stop = simulation_stop

    def load_logs(self, simulator_log_directory=None):
        simulator_log_directory = (
            self.simulator_log_directory
            if simulator_log_directory is None
            else simulator_log_directory
        )
        assert simulator_log_directory is not None
        # Read files in log dir and sort them by time
        archive_path = os.path.join(simulator_log_directory, "data.zip")
        with zipfile.ZipFile(archive_path, mode="r") as zf:
            files = sorted(zf.namelist())
            # Load 'em up
            logs = []
            for file in files:
                with zf.open(file, "r") as f:
                    logs.extend(pickle.load(f))
            return logs

    def get_encounter_logs(self, logs=None, simulator_log_directory=None):
        logs = self.load_logs(simulator_log_directory) if logs is None else logs
        encounter_logs = [log for log in logs if log["event_type"] == "encounter"]
        return encounter_logs

    @staticmethod
    def make_encounter_graph(encounter_logs):
        encounter_graph = nx.MultiDiGraph()
        for encounter_log in encounter_logs:
            # Ids
            human1_id = encounter_log["payload"]["unobserved"]["human1"]["human_id"]
            human2_id = encounter_log["payload"]["unobserved"]["human2"]["human_id"]
            # Write states
            human1_state = Dict()
            human2_state = Dict()
            # fmt: off
            # Infection status
            human1_state.is_infected = \
                (encounter_log["payload"]["unobserved"]["human1"]["infection_timestamp"]
                 is not None)
            human2_state.is_infected = \
                (encounter_log["payload"]["unobserved"]["human2"]["infection_timestamp"]
                 is not None)
            human1_state.got_exposed = \
                encounter_log["payload"]["unobserved"]["human1"]["got_exposed"]
            human2_state.got_exposed = \
                encounter_log["payload"]["unobserved"]["human2"]["got_exposed"]
            human1_state.caused_exposure = \
                encounter_log["payload"]["unobserved"]["human1"]["exposed_other"]
            human2_state.caused_exposure = \
                encounter_log["payload"]["unobserved"]["human2"]["exposed_other"]
            # Health and symptoms
            human1_state.symptoms = \
                encounter_log["payload"]["unobserved"]["human1"]["symptoms"]
            human2_state.symptoms = \
                encounter_log["payload"]["unobserved"]["human2"]["symptoms"]
            # Risk
            human1_state.risk = \
                encounter_log["payload"]["unobserved"]["human1"]["risk"]
            human2_state.risk = \
                encounter_log["payload"]["unobserved"]["human2"]["risk"]
            human1_state.risk_level = \
                encounter_log["payload"]["unobserved"]["human1"]["risk_level"]
            human2_state.risk_level = \
                encounter_log["payload"]["unobserved"]["human2"]["risk_level"]
            # Recommendations
            human1_state.rec_level = \
                encounter_log["payload"]["unobserved"]["human1"]["rec_level"]
            human2_state.rec_level = \
                encounter_log["payload"]["unobserved"]["human2"]["rec_level"]
            # fmt: on
            # Write stuff out
            encounter_graph.add_edge(
                human1_id,
                human2_id,
                encounter_log["time"],
                human1_state=human1_state,
                human2_state=human2_state,
            )
        return encounter_graph

    @staticmethod
    def filter_encounter_graph(graph, from_time, to_time):
        filtered = nx.MultiDiGraph()
        filtered.add_nodes_from(graph.nodes)
        for h1, h2, time in graph.edges:
            if from_time <= time <= to_time:
                filtered.add_edge(h1, h2, time, **graph.edges[h1, h2, time])
        return filtered

    @staticmethod
    def filter_edges(graph, condition_fn):
        assert callable(condition_fn)
        trimmed = nx.MultiDiGraph()
        for u, v, d in graph.edges:
            if condition_fn(d, **graph[u][v][d]):
                trimmed.add_edge(u, v, d, **graph[u][v][d])
        return trimmed

    @staticmethod
    def collate_edges(graph, collate_fn):
        trimmed = nx.MultiDiGraph()
        for u in graph.nodes:
            for v in graph[u]:
                new_d, new_attries = collate_fn(graph[u][v])
                trimmed.add_edge(u, v, new_d, **new_attries)
        return trimmed

    @staticmethod
    def to_digraph(graph, attry_extractor=(lambda human1_state, human2_state: {})):
        digraph = nx.DiGraph()
        for u in graph:
            for v in graph[u]:
                assert len(graph[u][v]) == 1
                time = list(graph[u][v].keys())[0]
                extra_attries = attry_extractor(**graph[u][v][time])
                digraph.add_edge(u, v, time=time, **extra_attries)
        return digraph

    @staticmethod
    def to_concat_digraph(
        graph, attry_extractor=(lambda human1_state, human2_state: {})
    ):
        digraph = nx.DiGraph()
        for u in graph:
            for v in graph[u]:
                attry_dicts = [
                    attry_extractor(**graph[u][v][time]) for time in graph[u][v].keys()
                ]
                concat_attry_dict = {
                    key: [d[key] for d in attry_dicts] for key in attry_dicts[0].keys()
                }
                concat_attry_dict.update({"time": list(graph[u][v].keys())})
                digraph.add_edge(u, v, **concat_attry_dict)
        return digraph

    # Edge filtering
    @staticmethod
    def is_contagion(time, human1_state, human2_state):
        return human2_state["got_exposed"]

    @staticmethod
    def is_contagion_without_symptoms(time, human1_state, human2_state):
        return (
            SimulationMetrics.is_contagion(time, human1_state, human2_state)
            and len(human1_state["symptoms"]) == 0
        )

    # Collating
    @staticmethod
    def keep_most_recent(edge_spec):
        most_recent_date = max(list(edge_spec.keys()))
        most_recent_attries = edge_spec[most_recent_date]
        return most_recent_date, most_recent_attries

    @staticmethod
    def keep_least_recent(edge_spec):
        least_recent_date = min(list(edge_spec.keys()))
        least_recent_attries = edge_spec[least_recent_date]
        return least_recent_date, least_recent_attries

    def compute_metrics(
        self,
        city=None,
        encounter_logs=None,
        logs=None,
        simulator_log_directory=None,
        simulation_start=None,
        simulation_stop=None,
        return_everything=False,
    ):
        if city is None:
            # Read in the logs
            encounter_logs = (
                encounter_logs
                if encounter_logs is not None
                else self.get_encounter_logs(logs, simulator_log_directory)
            )
        else:
            encounter_logs = self.get_encounter_logs(logs=city.events)
        # Build and filter the graph such that the encounters lie between sim start
        # and stop.
        simulation_start = (
            simulation_start if simulation_start is not None else self.simulation_start
        )
        simulation_stop = (
            simulation_stop if simulation_stop is not None else self.simulation_stop
        )
        encounter_graph = self.filter_encounter_graph(
            graph=self.make_encounter_graph(encounter_logs),
            from_time=simulation_start,
            to_time=simulation_stop,
        )
        # Get the contagion graph with only the ecounters that happened without
        # symptoms
        contagion_graph = self.filter_edges(
            encounter_graph, self.is_contagion_without_symptoms
        )
        _attry_extractor = lambda human1_state, human2_state: dict(
            exposer_risk=human1_state["risk"],
            exposer_risk_level=human1_state["risk_level"],
            exposer_rec_level=human1_state["rec_level"],
        )
        contagion_digraph = self.to_digraph(contagion_graph, _attry_extractor)
        encounter_digraph = self.to_concat_digraph(encounter_graph, _attry_extractor)
        # Read the risks from the graph
        raw_risks_all = [
            _risk
            for u, v in encounter_digraph.edges
            for _risk in encounter_digraph[u][v]["exposer_risk"]
        ]
        raw_risks_at_exposure = [
            contagion_digraph[u][v]["exposer_risk"] for u, v in contagion_digraph.edges
        ]
        metrics = self._evaluate_metrics_given_risks(
            raw_risks_all=raw_risks_all, raw_risks_at_exposure=raw_risks_at_exposure
        )
        if return_everything:
            everything = Dict()
            everything.encounter_graph = encounter_graph
            everything.contagion_graph = contagion_graph
            everything.encounter_digraph = encounter_digraph
            everything.contagion_digraph = contagion_digraph
            everything.raw_risks_all = raw_risks_all
            everything.raw_risks_at_exposure = raw_risks_at_exposure
            everything.metrics = metrics
            return everything
        else:
            return metrics

    @staticmethod
    def _evaluate_metrics_given_risks(raw_risks_all, raw_risks_at_exposure):
        thresholds = sorted(
            list(np.linspace(min(raw_risks_all), max(raw_risks_all) + 1 / 100, 100))
            + [np.percentile(raw_risks_at_exposure, x) for x in np.linspace(0, 100, 20)]
            + [np.percentile(raw_risks_all, x) for x in np.linspace(0, 100, 20)]
        )
        raw_risks_above_thresh = np.array(
            [(raw_risks_all >= thresh).mean() for thresh in thresholds]
        )
        exposure_risk_above_thresh = np.array(
            [(raw_risks_at_exposure >= thresh).mean() for thresh in thresholds]
        )
        output = Dict()
        output.raw_risks_above_thresh = raw_risks_above_thresh
        output.exposure_risk_above_thresh = exposure_risk_above_thresh
        output.thresholds = thresholds
        return output

    __call__ = compute_metrics
