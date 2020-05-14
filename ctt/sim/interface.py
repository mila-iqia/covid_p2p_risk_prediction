import os
import time
import persistqueue
import uuid
import shutil

import torch
from typing import TYPE_CHECKING, List
import ctt.sim.utils as su

if TYPE_CHECKING:
    # This is to trick pycharm in to giving me autocomplete for the mixin. ;)
    from ctt.training import CTTTrainer as Base
else:

    class Base(object):
        pass


class _SimInterfaceMixin(Base):
    def start_sim_server(self):
        pass

    def send_weights(self):
        return self

    def receive_metrics(self):
        return []


class DummySimInterfaceMixin(_SimInterfaceMixin):
    pass


class SimInterfaceMixin(_SimInterfaceMixin):
    @property
    def queue_directory(self) -> str:
        directory = su.get_queue_directory(self.experiment_directory)
        # Make a directory if it doesn't already exist
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        return directory

    @property
    def evaluation_checkpoint_directory(self) -> str:
        directory = su.get_evaluation_checkpoint_directory(self.experiment_directory)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        return directory

    @property
    def outgoing_queue(self) -> persistqueue.SQLiteQueue:
        if not hasattr(self, "_sim_interface_out_queue"):
            setattr(
                self,
                "_sim_interface_out_queue",
                persistqueue.SQLiteQueue(
                    self.queue_directory,
                    name=su.get_outgoing_queue_name(),
                    auto_commit=True,
                ),
            )
        return getattr(self, "_sim_interface_out_queue")

    @property
    def incoming_queue(self) -> persistqueue.SQLiteQueue:
        if not hasattr(self, "_sim_interface_in_queue"):
            setattr(
                self,
                "_sim_interface_in_queue",
                persistqueue.SQLiteQueue(
                    self.queue_directory,
                    name=su.get_incoming_queue_name(),
                    auto_commit=True,
                ),
            )
        return getattr(self, "_sim_interface_in_queue")

    def send_weights(self):
        # Dump weights
        weight_path = os.path.join(
            self.evaluation_checkpoint_directory, f"{uuid.uuid4().hex}.ckpt"
        )
        torch.save(self.model.state_dict(), weight_path)
        # Make payload to dump to the queue
        payload = dict(
            epoch=self.epoch,
            step=self.step,
            experiment_directory=self.experiment_directory,
            time=time.time(),
            weight_path=weight_path,
            simulation_kwargs=self.get("sim/kwargs", {})
        )
        self.outgoing_queue.put(payload)
        return self

    def receive_metrics(self) -> List[dict]:
        metrics = []
        while True:
            try:
                metrics.append(self.incoming_queue.get(block=False))
            except persistqueue.Empty:
                break
        return self.clear_weights(metrics)

    def clear_weights(self, metrics):
        for metric in metrics:
            weight_path = metric["weight_path"]
            if os.path.exists(weight_path):
                shutil.rmtree(weight_path)
        return metrics
