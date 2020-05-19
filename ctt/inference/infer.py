import os
from speedrun import BaseExperiment

from ctt.data_loading.loader import ContactPreprocessor
import ctt.models.transformer as tr
import torch
import torch.jit


class InferenceEngine(BaseExperiment):
    def __init__(self, experiment_directory, weight_path=None):
        super(InferenceEngine, self).__init__(experiment_directory=experiment_directory)
        self.record_args().read_config_file()
        self._build(weight_path=weight_path)

    def _build(self, weight_path=None):
        self.preprocessor = ContactPreprocessor(
            relative_days=self.get("data/loader_kwargs/relative_days", True),
            clip_history_days=self.get("data/loader_kwargs/clip_history_days", False),
            bit_encoded_messages=self.get(
                "data/loader_kwargs/bit_encoded_messages", True
            ),
        )
        self.model = self.load(weight_path=weight_path)

    def load(self, weight_path=None):
        path = (
            os.path.join(self.checkpoint_directory, "best.ckpt")
            if weight_path is None
            else weight_path
        )
        if path.endswith(".trace") or os.path.exists(path + ".trace"):
            if not path.endswith(".trace"):
                path += ".trace"  # load trace instead; inference should be faster
            model = torch.jit.load(path, map_location=torch.device("cpu"))
        else:
            assert os.path.exists(path)
            model_cls = getattr(tr, self.get("model/name", "ContactTracingTransformer"))
            model: torch.nn.Module = model_cls(
                **self.get("model/kwargs", {})
            )
            state = torch.load(path, map_location=torch.device("cpu"))
            model.load_state_dict(state["model"])
        model.eval()
        return model

    def infer(self, human_day_info):
        with torch.no_grad():
            model_input = self.preprocessor.preprocess(human_day_info, as_batch=True)
            model_output = self.model(model_input.to_dict())
            if isinstance(self.model, torch.jit.ScriptModule):
                # traced model outputs a tuple due to design limitation; remap here
                model_output = {
                    "encounter_variables": model_output[0],
                    "latent_variable": model_output[1],
                }
            contagion_proba = (
                model_output["encounter_variables"].sigmoid().numpy()[0, :, 0]
            )
            infectiousness = model_output["latent_variable"].numpy()[0, :, 0]
        return dict(contagion_proba=contagion_proba, infectiousness=infectiousness)
