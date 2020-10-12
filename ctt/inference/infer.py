import os
from speedrun import BaseExperiment

from ctt.data_loading.loader import ContactPreprocessor
from ctt.data_loading.transforms import get_transforms, get_pre_transforms, Transform
import ctt.models as tr
import torch
import torch.jit


class InferenceEngine(BaseExperiment):
    EXP_DIR_SEP = "@"

    def __init__(self, experiment_directory, weight_path=None, macro_path=None):
        """
        Parameters
        ----------
        experiment_directory : str
            Experiment directory to initialize the model and read weights from.
            Can either be:
                -> `path/to/experiment/directory`
                -> `path/to/experiment/directory@path/to/weights`
            In the latter case, `path/to/weights` is used as a substitute for the
            `weight_path` argument, if the latter is set to its default value of None.
        weight_path : str
            Path to weights. If left at None, the weight path can still be set
            by the `experiment_directory` string. If not left at None,
            this argument takes precedence.
        macro_path : str
            Path to macro, if applicable. Can be set by the env variable
            INFERENCE_ENGINE_MACRO.
        """
        experiment_directory_components = experiment_directory.split(self.EXP_DIR_SEP)
        # Init superclass with the first component, which is always the
        # experiment directory.
        super(InferenceEngine, self).__init__(
            experiment_directory=experiment_directory_components[0]
        )
        if weight_path is None and len(experiment_directory_components) > 1:
            weight_path = experiment_directory_components[1]
        self.record_args().read_config_file()
        macro_path = self._get_macro_path() if macro_path is None else macro_path
        if macro_path is not None:
            self.read_macro(macro_path)
        self._build(weight_path=weight_path)

    @staticmethod
    def _get_macro_path():
        return os.getenv("INFERENCE_ENGINE_MACRO", None)

    def _build(self, weight_path=None):
        test_transforms = get_transforms(self.get("data/transforms/test", {}))
        test_pretransforms = get_pre_transforms(self.get("data/pre_transforms", {}))
        self.preprocessor = ContactPreprocessor(
            relative_days=self.get("data/loader_kwargs/relative_days", True),
            clip_history_days=self.get("data/loader_kwargs/clip_history_days", False),
            bit_encoded_messages=self.get(
                "data/loader_kwargs/bit_encoded_messages", True
            ),
            mask_current_day_encounters=self.get(
                "data/loader_kwargs/mask_current_day_encounters", False
            ),
            transforms=test_transforms,
            pre_transforms=test_pretransforms,
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
            model: torch.nn.Module = model_cls(**self.get("model/kwargs", {}))
            state = torch.load(path, map_location=torch.device("cpu"))
            model.load_state_dict(state["model"])
        model.eval()
        return model

    def infer(self, human_day_info, return_full_output=False):
        with torch.no_grad():
            model_input = self.preprocessor.preprocess(human_day_info, as_batch=True)
            model_output = self.model(model_input.to_dict())
            if isinstance(self.model, torch.jit.ScriptModule):
                # traced model outputs a tuple due to design limitation; remap here
                model_output = {
                    "encounter_variables": model_output[0],
                    "latent_variable": model_output[1],
                }
            with Transform.invert_all_transforms():
                model_output = self.preprocessor.transforms(model_output)
            contagion_proba = (
                model_output["encounter_variables"].sigmoid().numpy()[0, :, 0]
            )
            # Nasim, don't you remember how bad unconditional squeezes effed you up
            # back in the days?
            infectiousness = model_output["latent_variable"].numpy()[0].squeeze()
        if not return_full_output:
            return dict(contagion_proba=contagion_proba, infectiousness=infectiousness)
        else:
            return dict(
                contagion_proba=contagion_proba,
                infectiousness=infectiousness,
                **model_output,
            )


def _profile(num_trials, experiment_directory, data_path):
    from ctt.data_loading.loader import ContactDataset
    import time

    print("Loading data...")
    dataset = ContactDataset(path=data_path)
    human_day_infos = [
        dataset.read(flat_idx=0) for k in range(num_trials)
    ]

    engine = InferenceEngine(experiment_directory)
    print(
        "This number should be 13: ",
        engine.model.health_profile_embedding[0].in_features,
    )
    _ = engine.infer(human_day_infos[0])

    print(f"Profiling {experiment_directory}...")
    start = time.time()
    for human_day_info in human_day_infos:
        _ = engine.infer(human_day_info)
    stop = time.time()

    print(f"Average time ({num_trials} trials): {(stop - start)/num_trials} s.")
