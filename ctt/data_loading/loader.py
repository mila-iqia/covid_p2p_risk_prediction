import pickle
from addict import Dict
import os
from typing import Union, List, TYPE_CHECKING
from copy import deepcopy
from contextlib import contextmanager
import zarr
import gc
import time

import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset, IterableDataset

from torch.utils.data.dataloader import DataLoader
from torch.nn.utils.rnn import pad_sequence

from ctt.data_loading.sampler import BinaryRejectionSampler


class InvalidSetSize(Exception):
    pass


class ContactDataset(Dataset):
    SET_VALUED_FIELDS = [
        "encounter_health",
        "encounter_message",
        "encounter_partner_id",
        "encounter_day",
        "encounter_duration",
        "encounter_is_contagion",
    ]

    SEQUENCE_VALUED_FIELDS = [
        "health_history",
        "infectiousness_history",
        "history_days",
        "valid_history_mask",
    ]

    DEFAULT_INPUT_FIELD_TO_SLICE_MAPPING = {
        "human_idx": ["human_idx", slice(None)],
        "day_idx": ["day_idx", slice(None)],
        "health_history": ["health_history", slice(None)],
        "reported_symptoms": ["health_history", slice(0, 27)],
        "test_results": ["health_history", slice(27, 28)],
        "age": ["health_profile", slice(0, 1)],
        "sex": ["health_profile", slice(1, 2)],
        "preexisting_conditions": ["health_profile", slice(2, 12)],
        "history_days": ["history_days", slice(None)],
        "valid_history_mask": ["valid_history_mask", slice(None)],
        "current_compartment": ["current_compartment", slice(None)],
        "infectiousness_history": ["infectiousness_history", slice(None)],
        "reported_symptoms_at_encounter": ["encounter_health", slice(0, 27)],
        "test_results_at_encounter": ["encounter_health", slice(27, 28)],
        "encounter_message": ["encounter_message", slice(None)],
        "encounter_partner_id": ["encounter_partner_id", slice(None)],
        "encounter_duration": ["encounter_duration", slice(None)],
        "encounter_day": ["encounter_day", slice(None)],
        "encounter_is_contagion": ["encounter_is_contagion", slice(None)],
    }

    TIME_VARYING_META_KEYS = {
        "hospitalization_per_day",
        "positive_test_results_per_day",
        "negative_test_results_per_day",
        "tested_per_day",
        "i_per_day",
    }

    # Compat with previous versions of the dataset
    # Age
    DEFAULT_AGE = 0
    ASSUMED_MAX_AGE = 100
    ASSUMED_MIN_AGE = 1
    AGE_NOT_AVAILABLE = 0
    # Sex
    DEFAULT_SEX = 0
    # Risk
    ASSUMED_MAX_RISK = 15
    ASSUMED_MIN_RISK = 0
    # Encounters
    DEFAULT_ENCOUNTER_DURATION = 10
    DEFAULT_PREEXISTING_CONDITIONS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Behaviour
    RAISE_IF_NO_ENCOUNTERS = False

    def __init__(
        self,
        path: str,
        relative_days=True,
        clip_history_days=False,
        bit_encoded_messages=False,
        forward_prediction=False,
        mask_current_day_encounters=False,
        transforms=None,
        pre_transforms=None,
        load_to_memory=False,
    ):
        """
        Parameters
        ----------
        path : str
            Path to the pickle file.
        relative_days : bool
            If set to True, the time-stamps (as days) are formatted such that
            the current day is represented as 0. Previous days are represented
            as negative values, i.e. day = -2 means the day before yesterday.
            If set to False, the time-stamps show the true number of days since
            day 0 (e.g. "today" can be represented as say 15).
        clip_history_days : bool
            Whether `history_days` (in the output) clips to the equivalent of
            day 0 for days that predate records. E.g. in day 5, whether
            `history_days[10]` is -5 (True) or -10 (False).
        bit_encoded_messages : bool
            Whether messages are encoded as a bit vector (True) or floats
            between 0 and 1 (False).
        forward_prediction : bool
            Whether to shift the infectiousness (i.e. target for the network) one
            step to the future.
        mask_current_day_encounters : bool
            Whether to mask out the encounters that happen at day zero (today).
        transforms : callable
            Transforms to apply before sending sample to the collator.
        pre_transforms : callable
            Transform that applies directly from to the data that is read
            from file, and before any further processing
        load_to_memory : bool
            Whether to load the dataset to memory or to read from file.
        """
        # Private
        self._num_id_bits = 16
        self._bit_encoded_age = False
        # Public
        self.path = path
        self.relative_days = relative_days
        self.clip_history_days = clip_history_days
        self.bit_encoded_messages = bit_encoded_messages
        self.forward_prediction = forward_prediction
        self.mask_current_day_encounters = mask_current_day_encounters
        self.transforms = transforms
        self.pre_transforms = pre_transforms
        self.load_to_memory = load_to_memory
        # Prepwork
        self._read_data()
        self._set_input_fields_to_slice_mapping()

    @property
    def hdf5_path(self):
        return os.path.join(self.path, "train.hdf5")

    @property
    def zarr_path(self):
        return os.path.join(self.path, "train.zarr")

    @property
    def meta_info_path(self):
        return os.path.join(self.path, "train_priors.pkl")

    @classmethod
    def is_dataset_path(cls, path: str):
        return (
            os.path.exists(path)
            and os.path.isdir(path)
            and os.path.exists(os.path.join(path, "train.zarr"))
            and os.path.exists(os.path.join(path, "train_priors.pkl"))
        )

    def _read_data(self):
        if self.path is not None:
            assert os.path.isdir(self.path), "Path must be a directory."
            assert os.path.exists(
                self.zarr_path
            ), f"Expecting a train.zarr in {self.path}, but found none."
            assert os.path.exists(
                os.path.join(self.path, "train_priors.pkl")
            ), f"Expecting a train_priors.pkl in {self.path}, but found none."
            zarr_store = zarr.open(self.zarr_path, "r")
            if self.load_to_memory:
                self._preloaded["dataset"] = zarr_store["dataset"][:]
            else:
                self._preloaded = zarr_store
            with open(self.meta_info_path, "rb") as f:
                self._meta_info = pickle.load(f)
            # This is an array of shape N3 where N is
            self._data_indices = np.array(
                np.asarray(self._preloaded["is_filled"]).nonzero()
            ).T
            self._num_days, _, self._num_humans = self._preloaded["dataset"].shape
        else:
            self._preloaded = None
            self._data_indices = np.zeros((0, 3))
            self._num_days, self._num_humans = 0, 0

    def _set_input_fields_to_slice_mapping(self):
        self._input_fields_to_slice_mapping = deepcopy(
            self.DEFAULT_INPUT_FIELD_TO_SLICE_MAPPING
        )
        # This method might grow.

    def load_in_memory(self):
        if isinstance(self._preloaded, zarr.hierarchy.Group):
            # Load up
            preloaded = {
                "dataset": self._preloaded["dataset"][:],
                "is_filled": self._preloaded["is_filled"][:],
            }
            # noinspection PyAttributeOutsideInit
            self._preloaded = preloaded
        else:
            # Make sure nothin' fishy
            assert isinstance(self._preloaded, dict)
            assert isinstance(self._preloaded["dataset"], np.ndarray)
            assert isinstance(self._preloaded["is_filled"], np.ndarray)
        return self

    def offload_from_memory(self):
        if isinstance(self._preloaded, zarr.hierarchy.Group):
            # Already offloaded
            pass
        else:
            # Replace preloaded with the zarr file
            assert isinstance(self._preloaded, dict)
            assert isinstance(self._preloaded["dataset"], np.ndarray)
            assert isinstance(self._preloaded["is_filled"], np.ndarray)
            del self._preloaded["dataset"]
            del self._preloaded["is_filled"]
            gc.collect()
            # noinspection PyAttributeOutsideInit
            self._preloaded = zarr.open(self.zarr_path, "r")

    @property
    def num_humans(self):
        return self._num_humans

    @property
    def num_days(self):
        return self._num_days

    def __len__(self):
        return self._data_indices.shape[0]

    def read(self, human_idx=None, day_idx=None, slot_idx=None, flat_idx=None):
        if flat_idx is not None:
            day_idx, slot_idx, human_idx = self._data_indices[flat_idx]
        try:
            human_day_info = self._preloaded["dataset"][day_idx, slot_idx, human_idx]
        except EOFError:
            raise ValueError(
                f"No stats found for human {human_idx} at day "
                f"{day_idx} and slot {slot_idx}."
            )
        human_day_info.update({"human_idx": human_idx, "slot_idx": slot_idx})
        return human_day_info

    def read_meta(self, human_idx=None, day_idx=None, flat_idx=None):
        if flat_idx is None:
            human_idx, _, day_idx = self._data_indices[flat_idx]
        time_varying_meta = {
            self._meta_info[key][day_idx] for key in self.TIME_VARYING_META_KEYS
        }
        human_meta = self._meta_info["humans"][f"human:{human_idx}"]
        return {"time_varying": time_varying_meta, "human": human_meta}

    def get(
        self,
        human_idx: int = None,
        day_idx: int = None,
        slot_idx: int = None,
        flat_idx: str = None,
        human_day_info: dict = None,
    ) -> Dict:
        """
        Parameters
        ----------
        human_idx : int
            Index specifying the human. Optional if file_name is provided.
        day_idx : int
            Index of the day. Optional if file_name is provided.
        slot_idx : int
            Index of the update slot of day. Optional if file_name is provided.
        flat_idx : str
            Index of the data-point. Optional if human_idx, day_idx
            and slot_idx is provided, but overrides the latter if provided.
        human_day_info : dict
            If provided, use this dictionary instead of the content of the
            pickle file (which is read from file).

        Returns
        -------
        Dict
            An addict with the following attributes:
                -> `health_history`: 14-day health history of self of shape (14, 28)
                        with channels `reported_symptoms` (27), `test_results`(1).
                -> `preexisting_conditions`: preexisting conditions reported by the
                    individual, of shape (10,).
                -> `age`: reported age of the individual of shape (1,). Note that age
                    is a float in the interval [0, 1] if reported, -1 otherwise.
                -> `sex`: reported sex of the individual of shape (1,). Note that sex
                    can be one of {-1, 0, 1, 2}, where:
                        -1: unreported
                         0: not known
                         1: female
                         2: male
                -> `human_idx`: the ID of the human individual, of shape (1,). If not
                    available, it's set to -1.
                -> `day_idx`: the day from which the sample originates, of shape (1,).
                -> `history_days`: time-stamps to go with the health_history,
                    of shape (14, 1).
                -> `valid_history_mask`: 1 if the time-stamp corresponds to a valid
                    point in history, 0 otherwise, of shape (14,).
                -> `current_compartment`: current epidemic compartment (S/E/I/R)
                    of shape (4,).
                -> `infectiousness_history`: 14-day history of infectiousness,
                    of shape (14, 1).
                -> `encounter_health`: health during encounter, of shape (M, 28)
                -> `encounter_message`: risk transmitted during encounter,
                        of shape (M, 8) if `self.bit_encoded_messages` is set to True,
                        of shape (M, 1) otherwise.
                        Bit encoded messages means that the integer is represented
                        as their corresponding bit vector.
                -> `encounter_partner_id`: id of the other in the encounter,
                        of shape (M, num_id_bits). If num_id_bits = 16, it means that the
                        id (say 65535) is represented in 16-bit binary.
                -> `encounter_duration`: duration of encounter, of shape (M, 1)
                -> `encounter_day`: the day of encounter, of shape (M, 1)
                -> `encounter_is_contagion`: whether the encounter was a contagion,
                    of shape (M, 1).
        """
        if human_day_info is None:
            human_day_info = self.read(human_idx, day_idx, slot_idx, flat_idx)
            if self.forward_prediction:
                raise NotImplementedError
            else:
                # Future is not needed, so we don't waste time loading it
                future_human_day_info = None
        else:
            # If we're here, we're running in the inference server -- future is
            # not available.
            future_human_day_info = None
        day_idx = human_day_info["current_day"]
        if human_idx is None:
            human_idx = human_day_info.get("human_idx", None)
        else:
            assert human_idx == human_day_info["human_idx"]
        if human_idx is None:
            human_idx = -1
        # Apply any pre-transforms
        if self.pre_transforms is not None:
            human_day_info = self.pre_transforms(human_day_info, human_idx, day_idx)
        # -------- Encounters --------
        # Extract info about encounters
        #   encounter_info.shape = M3, where M is the number of encounters.
        encounter_info = human_day_info["observed"]["candidate_encounters"]
        if encounter_info.size == 0:
            encounter_info = encounter_info.reshape(0, 4)
        num_encounters = encounter_info.shape[0]
        if num_encounters == 0 and self.RAISE_IF_NO_ENCOUNTERS:
            raise InvalidSetSize
        if num_encounters > 0:
            valid_encounter_mask = encounter_info[:, 3] > (day_idx - 14)
            if self.mask_current_day_encounters:
                valid_encounter_mask = np.logical_and(
                    valid_encounter_mask, encounter_info[:, 3] != day_idx
                )
            encounter_info = encounter_info[valid_encounter_mask]
            # The number of valid encounters might have changed after the masking.
            num_encounters = encounter_info.shape[0]
        else:
            valid_encounter_mask = np.ones((0,), dtype="bool")
        assert encounter_info.shape[1] == 4
        (
            encounter_partner_id,
            encounter_message,
            encounter_duration,
            encounter_day,
        ) = (
            encounter_info[:, 0],
            encounter_info[:, 1],
            encounter_info[:, 2],
            encounter_info[:, 3],
        )
        if num_encounters > 0:
            # Convert partner-id's to binary (shape = (M, num_id_bits))
            encounter_partner_id = (
                np.unpackbits(
                    encounter_partner_id.astype(f"uint{self._num_id_bits}").view(
                        "uint8"
                    )
                )
                .reshape(num_encounters, self._num_id_bits)
                .astype("float32")
            )
        else:
            encounter_partner_id = np.zeros((0, self._num_id_bits)).astype("float32")
        # Convert risk
        encounter_message = self._fetch_encounter_message(
            encounter_message, num_encounters
        )
        encounter_is_contagion = self._fetch_encounter_is_contagion(
            human_day_info, valid_encounter_mask, encounter_day
        )
        encounter_day = encounter_day.astype("float32")
        # -------- Health --------
        # Get health info
        health_history = self._fetch_health_history(human_day_info)
        infectiousness_history, mask_head = self._fetch_infectiousness_history(
            human_day_info, future_human_day_info
        )
        viral_load_history, vl2i_multiplier = self._fetch_viral_load_history(
            infectiousness_history, human_day_info
        )
        exposure_history = self._fetch_exposure_history(human_day_info)
        history_days = np.arange(day_idx - 13, day_idx + 1)[::-1, None]
        valid_history_mask = (history_days >= 0)[:, 0]
        if mask_head:
            # Mask out the head (because we're "out of future" to read from)
            valid_history_mask[0] = 0
        # Get historical health info given the day of encounter (shape = (M, 13))
        if num_encounters > 0:
            encounter_at_historical_day_idx = np.argmax(
                encounter_day == history_days, axis=0
            )
            health_at_encounter = health_history[encounter_at_historical_day_idx, :]
        else:
            health_at_encounter = np.zeros(
                (0, health_history.shape[-1]), dtype=health_history.dtype
            )
        # TODO: Get prevalence-related variables at the time of encounter
        # Get current epidemiological compartment
        currently_infected = (
            infectiousness_history[(0 if not self.forward_prediction else 1), 0] > 0.0
        )
        if human_day_info["unobserved"]["is_recovered"]:
            current_compartment = "R"
        elif currently_infected:
            current_compartment = "I"
        elif human_day_info["unobserved"]["is_exposed"]:
            current_compartment = "E"
        else:
            current_compartment = "S"
        current_compartment = np.array(
            [
                current_compartment == "S",
                current_compartment == "E",
                current_compartment == "I",
                current_compartment == "R",
            ]
        ).astype("float32")
        # Get age and sex if available, else use a default
        age = self._fetch_age(human_day_info)
        sex = np.array([human_day_info["observed"].get("sex", self.DEFAULT_SEX)])
        preexsting_conditions = human_day_info["observed"].get(
            "preexisting_conditions", np.array(self.DEFAULT_PREEXISTING_CONDITIONS)
        )
        health_profile = np.concatenate([age, sex, preexsting_conditions])
        # Clip history days if required
        if self.clip_history_days:
            history_days = np.clip(history_days, 0, None)
        # Normalize both days to assign 0 to present
        if self.relative_days:
            history_days = history_days - day_idx
            encounter_day = encounter_day - day_idx
        # This should be it
        sample = Dict(
            human_idx=torch.from_numpy(np.array([human_idx])),
            day_idx=torch.from_numpy(np.array([day_idx])),
            health_history=torch.from_numpy(health_history).float(),
            health_profile=torch.from_numpy(health_profile).float(),
            preexsting_conditions=torch.from_numpy(preexsting_conditions).float(),
            age=torch.from_numpy(age).float(),
            sex=torch.from_numpy(sex).float(),
            infectiousness_history=torch.from_numpy(infectiousness_history).float(),
            viral_load_history=torch.from_numpy(viral_load_history).float(),
            vl2i_multiplier=torch.from_numpy(vl2i_multiplier).float(),
            exposure_history=torch.from_numpy(exposure_history).float(),
            history_days=torch.from_numpy(history_days).float(),
            valid_history_mask=torch.from_numpy(valid_history_mask).float(),
            current_compartment=torch.from_numpy(current_compartment).float(),
            encounter_health=torch.from_numpy(health_at_encounter).float(),
            encounter_message=torch.from_numpy(encounter_message).float(),
            encounter_partner_id=torch.from_numpy(encounter_partner_id).float(),
            encounter_day=torch.from_numpy(encounter_day[:, None]).float(),
            encounter_duration=torch.from_numpy(encounter_duration[:, None]).float(),
            encounter_is_contagion=torch.from_numpy(encounter_is_contagion).float(),
        )
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    def _fetch_age(self, human_day_info):
        age = human_day_info["observed"].get("age", self.DEFAULT_AGE)
        if self._bit_encoded_age:
            if age == -1:
                age = np.array([-1] * 8).astype("int")
            else:
                age = np.unpackbits(np.array([age]).astype("uint8")).astype("int")
        else:
            if age == -1:
                age = np.array([-1.0])
            else:
                age = (age - self.ASSUMED_MIN_AGE) / (
                    self.ASSUMED_MAX_AGE - self.ASSUMED_MIN_AGE
                )
                age = np.array([age])
        return age

    def _fetch_encounter_is_contagion(
        self, human_day_info, valid_encounter_mask, encounter_day
    ):
        if valid_encounter_mask.shape[0] == 0:
            # Empty tensor
            return np.zeros((0, 1)).astype("float32")
        else:
            return human_day_info["unobserved"]["exposure_encounter"][
                valid_encounter_mask, None
            ].astype("float32")

    def _fetch_infectiousness_history(self, human_day_info, future_human_day_info=None):
        mask_head = False
        if self.forward_prediction:
            if future_human_day_info is not None:
                # We're predicting one step in the future, so we simply use
                # the infectiousness history from one step in the future.
                infectiousness_history = future_human_day_info["unobserved"][
                    "infectiousness"
                ]
            else:
                # We want to predict one step in the future, but this future is not
                # available. So we shift the current history one step to the future,
                # and repeat the current infectiousness as a placeholder for future
                # infectiousness. This will be masked downstream.
                mask_head = True
                _infectiousness_history = human_day_info["unobserved"]["infectiousness"]
                infectiousness_history = np.concatenate(
                    [_infectiousness_history[0:1], _infectiousness_history[:-1]], axis=0
                )
        else:
            # No forward prediction, plain old retrospective prediction
            infectiousness_history = human_day_info["unobserved"]["infectiousness"]
        assert infectiousness_history.ndim == 1
        if infectiousness_history.shape[0] < 14:
            infectiousness_history = np.pad(
                infectiousness_history,
                ((0, 14 - infectiousness_history.shape[0]),),
                mode="constant",
            )
        assert infectiousness_history.shape[0] == 14
        return infectiousness_history[:, None], mask_head

    def _fetch_viral_load_history(self, infectiousness_history, human_day_info):
        multiplier = human_day_info["unobserved"][
            "viral_load_to_infectiousness_multiplier"
        ]
        if multiplier is not None:
            viral_load_history = infectiousness_history / multiplier
            multiplier = np.array([multiplier]).astype("float32")
        else:
            assert infectiousness_history.sum() == 0, (
                "Human is infectious but `viral_load_to_infectiousness_multiplier`"
                " is None."
            )
            viral_load_history = infectiousness_history
            multiplier = np.array([0.0]).astype("float32")
        return viral_load_history, multiplier

    def _fetch_exposure_history(self, human_day_info):
        exposed_since = human_day_info["unobserved"]["exposure_day"]
        exposure_history = np.zeros(shape=(14,))
        if exposed_since is not None and exposed_since < 14:
            exposure_history[exposed_since] = 1
        return exposure_history[:, None]

    def _fetch_encounter_message(self, encounter_message, num_encounters):
        if self.bit_encoded_messages:
            if encounter_message.shape[0] == 0:
                # Empty message tensor
                return np.zeros((0, 8))
            else:
                # Convert to bit-vector
                return (
                    np.unpackbits(encounter_message.astype("uint8"))
                    .reshape(num_encounters, -1)
                    .astype("float32")
                )
        else:
            if encounter_message.shape[0] == 0:
                # Empty message tensor
                return np.zeros((0, 1))
            else:
                # max-min normalize message
                return (
                    (encounter_message[:, None] - self.ASSUMED_MIN_RISK)
                    / (self.ASSUMED_MAX_RISK - self.ASSUMED_MIN_RISK)
                ).astype("float32")

    def _fetch_health_history(self, human_day_info):
        if human_day_info["observed"]["reported_symptoms"].shape[1] == 27:
            symptoms = human_day_info["observed"]["reported_symptoms"]
        elif human_day_info["observed"]["reported_symptoms"].shape[1] == 28:
            # Remove the mystery symptom
            symptoms = human_day_info["observed"]["reported_symptoms"][:, :-1]
        else:
            raise ValueError
        return np.concatenate(
            [symptoms, human_day_info["observed"]["test_results"][:, None],], axis=1,
        )

    def _fetch_prevalence_history(self, day_idx):
        raise NotImplementedError

    def __getitem__(self, item):
        return self.get(flat_idx=item)

    @classmethod
    def collate_fn(cls, batch):
        fixed_size_collates = {
            key: torch.stack([x[key] for x in batch], dim=0)
            for key in batch[0].keys()
            if key not in cls.SET_VALUED_FIELDS
        }

        # Make a mask
        max_set_len = max([x[cls.SET_VALUED_FIELDS[0]].shape[0] for x in batch])
        set_lens = torch.tensor([x[cls.SET_VALUED_FIELDS[0]].shape[0] for x in batch])
        mask = (
            torch.arange(max_set_len, dtype=torch.long)
            .expand(len(batch), max_set_len)
            .lt(set_lens[:, None])
        ).float()
        # Pad the set elements by writing in place to pre-made tensors
        padded_collates = {
            key: pad_sequence([x[key] for x in batch], batch_first=True)
            for key in cls.SET_VALUED_FIELDS
        }
        # Make the final addict and return
        collates = Dict(mask=mask)
        collates.update(fixed_size_collates)
        collates.update(padded_collates)
        return collates

    @staticmethod
    def extract(
        cls_or_self,
        tensor_or_dict: Union[torch.Tensor, dict],
        query_field: str,
        tensor_name: str = None,
    ) -> torch.Tensor:
        """
        This function can do two things.
            1. Given a dict (output from __getitem__/get or from collate_fn),
               extract the field given by name `query_fields`.
            2. Given a tensor and a `tensor_name`, assume that the tensor originated
               by indexing the dictionary returned by `__getitem__`/`get`/`collate_fn`
               with `tensor_name`. Now, proceed to extract the field given by name
               `query_fields`.
        Parameters
        ----------
        cls_or_self: type or ContactDataset
            Either an instance (self) or the class.
        tensor_or_dict : torch.Tensor or dict
            Torch tensor or dictionary.
        query_field : str
            Name of the field to extract.
        tensor_name : str
            If `tensor_or_dict` is a torch tensor, assume this is the dictionary
            key that was used to obtain the said tensor. Can be set to None if
            tensor_or_dict is a dict, but if not, it will be validated.

        Returns
        -------
        torch.Tensor
        """
        if isinstance(cls_or_self, type) and issubclass(cls_or_self, ContactDataset):
            mapping = cls_or_self.DEFAULT_INPUT_FIELD_TO_SLICE_MAPPING
        elif isinstance(cls_or_self, ContactDataset):
            mapping = cls_or_self._input_fields_to_slice_mapping
        else:
            raise TypeError
        return cls_or_self._extract(mapping, tensor_or_dict, query_field, tensor_name,)

    @staticmethod
    def _extract(
        field_to_slice_mapping: dict,
        tensor_or_dict: Union[torch.Tensor, dict],
        query_field: str,
        tensor_name: str = None,
    ) -> torch.Tensor:
        assert query_field in field_to_slice_mapping
        if isinstance(tensor_or_dict, dict):
            tensor_name, slice_ = field_to_slice_mapping[query_field]
            tensor = tensor_or_dict[tensor_name]
        elif torch.is_tensor(tensor_or_dict):
            target_tensor_name, slice_ = field_to_slice_mapping[query_field]
            if tensor_name is not None:
                assert target_tensor_name == tensor_name
            tensor = tensor_or_dict
        else:
            raise TypeError
        return tensor[..., slice_]

    def __del__(self):
        if self._preloaded is not None and hasattr(self._preloaded, "close"):
            self._preloaded.close()


class ContactDatastream(IterableDataset):
    def __init__(
        self,
        datasets: List[ContactDataset],
        shuffle_in_dataset: bool = False,
        rejection_sampler: "BinaryRejectionSampler" = None,
        base_seed: int = None,
    ):
        self.datasets = list(datasets)
        self.shuffle_in_dataset = shuffle_in_dataset
        self.rejection_sampler = rejection_sampler
        self.base_seed = base_seed or np.random.randint(0, 1000000)
        self._seed = base_seed
        self._epoch_num = None

    def _iter(self):
        for dataset in self.datasets:
            with self.in_memory(dataset):
                if self.shuffle_in_dataset:
                    idxs = np.random.permutation(len(dataset))
                else:
                    idxs = range(len(dataset))
                for idx in idxs:
                    sample = dataset[idx]
                    if self.rejection_sampler is not None:
                        sample = self.rejection_sampler(sample)
                    if sample is None:
                        continue
                    else:
                        yield sample

    def set_epoch(self, epoch):
        self._epoch_num = epoch
        return self

    def auto_seed(self, worker_id):
        # Seed derives from worker-id and the epoch.
        epoch_num = self._epoch_num or (int(time.time() * 10000000) % 10000000)
        self._seed = (epoch_num * 10000 + worker_id) % 4294967295
        if self.rejection_sampler is not None:
            self.rejection_sampler.seed(self._seed)

    def __iter__(self):
        return self._iter()

    def __len__(self):
        return sum([len(dataset) for dataset in self.datasets])

    @staticmethod
    @contextmanager
    def in_memory(dataset):
        dataset.load_in_memory()
        yield
        dataset.offload_from_memory()

    @staticmethod
    def worker_init_fn(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers
        worker_id = worker_info.id
        self = worker_info.dataset
        self.datasets = self.datasets[worker_id::num_workers]
        self.auto_seed(worker_id)


class ContactPreprocessor(ContactDataset):
    def __init__(self, **kwargs):
        # noinspection PyTypeChecker
        super(ContactPreprocessor, self).__init__(path=None, **kwargs)
        self._num_humans = 1
        self._num_days = 1
        self._preloaded = None

    def _read_data(self):
        # Defuse this method since it's not needed anymore
        self._day_idx_offset = 0
        self._human_idx_offset = 1

    def preprocess(self, human_day_info, as_batch=True):
        # noinspection PyTypeChecker
        sample = self.get(None, None, None, human_day_info=human_day_info)
        if as_batch:
            sample = self.collate_fn([sample])
        return sample

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError


class EpochCountingDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(EpochCountingDataLoader, self).__init__(*args, **kwargs)
        self._epoch_count = 0

    def __iter__(self):
        if hasattr(self.dataset, "set_epoch"):
            getattr(self.dataset, "set_epoch")(self._epoch_count)
        rval = super(EpochCountingDataLoader, self).__iter__()
        self._epoch_count += 1
        return rval


def get_dataloader(
    batch_size,
    shuffle=True,
    num_workers=1,
    rng=None,
    stream=False,
    rejection_sampler_kwargs=None,
    **dataset_kwargs,
):
    path = dataset_kwargs.pop("path")
    num_datasets_to_select = dataset_kwargs.pop("num_datasets_to_select", None)
    worker_init_fn = None
    if isinstance(path, str):
        if not ContactDataset.is_dataset_path(path):
            # This code-path supports the case where path is a directory of zip files.
            # If `num_datasets_to_select` is None, then all zips in the directory are
            # selected.
            paths = [
                p
                for p in os.listdir(path)
                if ContactDataset.is_dataset_path(os.path.join(path, p))
            ]
            assert len(paths) > 0, f"No dataset paths found in directory: {path}"
            if num_datasets_to_select is not None:
                rng = np.random.RandomState() if rng is None else rng
                paths = rng.choice(
                    paths,
                    num_datasets_to_select,
                    replace=num_datasets_to_select > len(paths),
                ).tolist()
            dataset = []
            for p in paths:
                try:
                    print(f"Reading dataset: {p}")
                    dataset.append(
                        ContactDataset(path=os.path.join(path, p), **dataset_kwargs)
                    )
                except OSError as e:
                    print(
                        f"Failed to read dataset at location "
                        f"{p} due to exception:\n{str(e)}"
                    )
            if stream:
                dataset = ContactDatastream(dataset, shuffle_in_dataset=shuffle)
                shuffle = False
                worker_init_fn = ContactDatastream.worker_init_fn
            else:
                dataset = ConcatDataset(dataset)
        else:
            # This codepath supports the case where path points to a zip.
            dataset = ContactDataset(path=path, **dataset_kwargs)
    elif isinstance(path, (list, tuple)):
        # This codepath supports the case where path is a list of paths pointing
        # to zips.
        assert all([ContactDataset.is_dataset_path(p) for p in path])
        dataset = [ContactDataset(path=p, **dataset_kwargs) for p in path]
        if stream:
            dataset = ContactDatastream(dataset, shuffle_in_dataset=shuffle)
            shuffle = False
            worker_init_fn = ContactDatastream.worker_init_fn
        else:
            dataset = ConcatDataset(
                [ContactDataset(path=p, **dataset_kwargs) for p in path]
            )
    else:
        raise TypeError
    if rejection_sampler_kwargs is not None:
        assert stream, "Rejection sampler is only supported for the streaming dataset."
        dataset.rejection_sampler = BinaryRejectionSampler(**rejection_sampler_kwargs)
    dataloader = EpochCountingDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=ContactDataset.collate_fn,
        worker_init_fn=worker_init_fn,
    )
    return dataloader
