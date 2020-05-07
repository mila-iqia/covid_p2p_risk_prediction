from typing import TYPE_CHECKING
import persistqueue

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
        pass

    def read_metrics_from_sim_server(self):
        pass


class DummySimInterfaceMixin(_SimInterfaceMixin):
    pass


class SimInterfaceMixin(_SimInterfaceMixin):
    
    def send_weights_to_sim_server(self):
        pass
    pass