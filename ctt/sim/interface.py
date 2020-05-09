import os
import time
import persistqueue

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # This is to trick pycharm in to giving me autocomplete for the mixin. ;)
    from ctt.training import CTTTrainer as Base
else:

    class Base(object):
        pass


class _SimInterfaceMixin(Base):
    def start_sim_server(self):
        pass

    def send_weights_to_sim_server(self):
        return self

    def receive_metrics_from_sim_server(self):
        return []


class DummySimInterfaceMixin(_SimInterfaceMixin):
    pass


class SimInterfaceMixin(_SimInterfaceMixin):
    @property
    def queue_directory(self) -> str:
        return os.path.join(self.experiment_directory, "Logs", "SimQ")

    @property
    def queue(self) -> persistqueue.SQLiteQueue:
        if not hasattr(self, "_sim_interface_queue"):
            setattr(
                self,
                "_sim_interface_queue",
                persistqueue.SQLiteQueue(
                    self.queue_directory, name="simq", auto_commit=True
                ),
            )
        return getattr(self, "_sim_interface_queue")

    def send_weights_to_sim_server(self):
        payload = dict(
            epoch=self.epoch,
            step=self.step,
            experiment_directory=self.experiment_directory,
            time=time.time(),
            weights=self.model.state_dict(),
        )
        self.queue.put(payload)
        return self

    def receive_metrics_from_sim_server(self):
        metrics = []
        while True:
            try:
                metrics.append(self.queue.get(block=False))
            except persistqueue.Empty:
                break
        return metrics
