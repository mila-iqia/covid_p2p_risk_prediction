import os
import zipfile
import pickle

from addict import Dict
import networkx as nx


class SimulationMetrics(object):
    def __init__(self, simulator_log_directory):
        self.simulator_log_directory = simulator_log_directory

    def load_logs(self):
        # Read files in log dir and sort them by time
        archive_path = os.path.join(self.simulator_log_directory, "data.zip")
        with zipfile.ZipFile(archive_path, mode="r") as zf:
            files = sorted(zf.namelist())
            # Load 'em up
            logs = []
            for file in files:
                with zf.open(file, "r") as f:
                    logs.extend(pickle.load(f))
            return logs

    def get_encounter_logs(self):
        logs = self.load_logs()
        encounter_logs = [log for log in logs if log["event_type"] == "encounter"]
        return encounter_logs

    def make_encounter_graph(self, encounter_logs=None):
        encounter_logs = encounter_logs or self.get_encounter_logs()
        encounter_graph = nx.MultiDiGraph()
        for encounter_log in encounter_logs:
            # Ids
            human1_id = encounter_log["payload"]["unobserved"]["human1"]["human_id"]
            human2_id = encounter_log["payload"]["unobserved"]["human2"]["human_id"]
            # Write states
            human1_state = Dict()
            human2_state = Dict()
            # Infection status
            human1_state.is_infected = (
                encounter_log["payload"]["unobserved"]["human1"]["infection_timestamp"]
                is not None
            )
            human2_state.is_infected = (
                encounter_log["payload"]["unobserved"]["human2"]["infection_timestamp"]
                is not None
            )
            human1_state.got_exposed = encounter_log["payload"]["unobserved"]["human1"][
                "got_exposed"
            ]
            human2_state.got_exposed = encounter_log["payload"]["unobserved"]["human2"][
                "got_exposed"
            ]
            human1_state.caused_exposure = encounter_log["payload"]["unobserved"][
                "human1"
            ]["exposed_other"]
            human2_state.caused_exposure = encounter_log["payload"]["unobserved"][
                "human2"
            ]["exposed_other"]
            # Health and symptoms
            human1_state.is_infected = encounter_log["payload"]["unobserved"]["human1"][
                "symptoms"
            ]
            human2_state.is_infected = encounter_log["payload"]["unobserved"]["human2"][
                "symptoms"
            ]
            # Risk
            human1_state.risk = encounter_log["payload"]["unobserved"]["human1"]["risk"]
            human2_state.risk = encounter_log["payload"]["unobserved"]["human2"]["risk"]
            human1_state.risk_level = encounter_log["payload"]["unobserved"]["human1"][
                "risk_level"
            ]
            human2_state.risk_level = encounter_log["payload"]["unobserved"]["human2"][
                "risk_level"
            ]
            # Recommendations
            human1_state.rec_level = encounter_log["payload"]["unobserved"]["human1"][
                "rec_level"
            ]
            human2_state.rec_level = encounter_log["payload"]["unobserved"]["human2"][
                "rec_level"
            ]
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
