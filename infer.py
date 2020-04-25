import os
from speedrun import BaseExperiment

from loader import ContactPreprocessor
from models import ContactTracingTransformer
import torch


class InferenceEngine(BaseExperiment):
    def __init__(self, experiment_directory):
        super(InferenceEngine, self).__init__(experiment_directory=experiment_directory)
        self.record_args().read_config_file()
        self._build()

    def _build(self):
        self.preprocessor = ContactPreprocessor(
            relative_days=self.get("data/loader_kwargs/relative_days", True)
        )
        self.model: torch.nn.Module = ContactTracingTransformer(
            **self.get("model/kwargs", {})
        )
        self.load()

    def load(self):
        path = os.path.join(self.checkpoint_directory, "best.ckpt")
        assert os.path.exists(path)
        state = torch.load(path)
        self.model.load_state_dict(state["model"])
        return self

    def infer(self, human_day_info):
        with torch.no_grad():
            model_input = self.preprocessor.preprocess(human_day_info, as_batch=True)
            model_output = self.model(model_input.to_dict())
            contagion_proba = (
                model_output["encounter_variables"].sigmoid().numpy()[0, :, 0]
            )
            infectiousness = model_output["latent_variable"].numpy()[0, :, 0]
        return dict(contagion_proba=contagion_proba, infectiousness=infectiousness)


if __name__ == "__main__":
    from loader import ContactDataset

    dataset = ContactDataset(
        path="/Users/nrahaman/Python/ctt/data/sim_people-1000_days-60_init-0"
    )
    hdi = dataset.read(0, 0)

    engine = InferenceEngine("/Users/nrahaman/Python/ctt/exp/DEBUG-0")
    output = engine.infer(hdi)
    pass
