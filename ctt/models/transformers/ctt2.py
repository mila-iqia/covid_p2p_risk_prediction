from typing import Union
from itertools import zip_longest

import torch
import torch.nn as nn

from ctt.models.transformers.ctt1 import _DiurnalContactTracingTransformer
import ctt.models.modules as mods
import ctt.models.attn as attn


class _DiurnalContactTracingTransformerV2(_DiurnalContactTracingTransformer):
    def __init__(
        self,
        *,
        health_history_embedding: nn.Module,
        health_profile_embedding: nn.Module,
        time_embedding: nn.Module,
        duration_embedding: nn.Module,
        partner_id_embedding: Union[nn.Module, None],
        message_embedding: nn.Module,
        encounter_pooler: nn.Module,
        post_pooling_mlp: nn.Module,
        attention_blocks: nn.ModuleList,
        latent_variable_mlp: nn.Module,
        encounter_mlp: nn.Module,
        entity_masker: nn.Module,
        partner_id_placeholder: nn.Parameter,
    ):
        nn.Module.__init__(self)
        # Privates
        self._diagnose = False
        self._output_as_tuple = False
        # Publics
        self.health_history_embedding = health_history_embedding
        self.health_profile_embedding = health_profile_embedding
        self.time_embedding = time_embedding
        self.duration_embedding = duration_embedding
        self.partner_id_embedding = partner_id_embedding
        self.message_embedding = message_embedding
        self.encounter_pooler = encounter_pooler
        self.post_pooling_mlp = post_pooling_mlp
        self.attention_blocks = attention_blocks
        self.latent_variable_mlp = latent_variable_mlp
        self.encounter_mlp = encounter_mlp
        self.entity_masker = entity_masker
        self.partner_id_placeholder = partner_id_placeholder

    def forward(self, inputs: dict) -> Union[dict, tuple]:
        """
        inputs is a dict containing the below keys. The format of the tensors
        are indicated as e.g. `BTC`, `BMC` (etc), which can be interpreted as
        following.
            B: batch size,
            T: size of the rolling window over health history (i.e. number of
               time-stamps),
            C: number of generic channels,
            M: number of encounters,
        Elements with pre-determined shapes are indicated as such.
        For example:
            - B(14) indicates a tensor of shape (B, 14),
            - BM1 indicates a tensor of shape (B, M, 1)
            - B(T=14)C indicates a tensor of shape (B, 14, C) where 14
                is the currently set size of the rolling window.

        Parameters
        ----------
        inputs : dict
            A python dict with the following keys:
                -> `health_history`: a B(T=14)C tensor of the 14-day health
                    history (symptoms + test results + day) of the individual.
                -> `health_profile`: a BC tensor of the health profile
                    containing (age + health + preexisting_conditions) of the
                    individual.
                -> `history_days`: a B(T=14)1 tensor of the day corresponding to the
                    T dimension in `health_history`.
                -> `encounter_health`: a BMC tensor of health during an
                    encounter indexed by M.
                -> `encounter_message`: a BMC tensor of the received
                    message from the encounter partner.
                -> `encounter_day`: a BM1 tensor of the encounter day.
                -> `encounter_duration`: a BM1 tensor of the encounter duration.
                    This is not the actual duration, but a proxy (for the number
                    of encounters)
                -> `encounter_partner_id`: a binary  BMC tensor specifying
                    the ID of the encounter partner.
                -> `mask`: a BM mask tensor distinguishing the valid entries (1)
                    from padding (0) in the set valued inputs.
                -> `valid_history_mask`: a B(14) mask tensor distinguising valid
                    points in history (1) from padding (0).
        Returns
        -------
        dict
            A dict containing the keys "encounter_variables" and "latent_variable".
        """
        # -------- Shape Wrangling --------
        batch_size = inputs["health_history"].shape[0]
        num_history_days = inputs["health_history"].shape[1]
        num_encounters = inputs["encounter_health"].shape[1]
        # -------- Embeddings --------
        embeddings = self.embed(inputs)
        # -------- Attentions --------
        # Get the daily and encounter-wise entities
        extracted_entities = self.extract_entities(inputs, embeddings)
        # Assemble the attention masks. The day-day coupling mask is pretty simple
        # and as expected:
        day_day_coupling_mask = (
            inputs["valid_history_mask"][:, :, None]
            * inputs["valid_history_mask"][:, None, :]
        )
        # But the coupling between day and encounter is slightly more complicated.
        # Recall that we want to pool together encounters between the same day.
        # The mask is therefore a tensor of size BTM, and is 1 if the encounter of
        # idx m/M happened at day of index t/T (in batch of index b/B).
        assert inputs["history_days"].shape[-1] == 1
        # BT1,BM1 -> BT,BM ->BTM
        day_encounter_coupling_mask = torch.eq(
            inputs["history_days"], inputs["encounter_day"][:, None, :, 0],
        ).float()
        # Additionally, we mask out the invalid entries
        day_encounter_coupling_mask = day_encounter_coupling_mask * (
            inputs["valid_history_mask"][:, :, None] * inputs["mask"][:, None, :]
        )
        # Attenzione!
        # Pool encounter entities to get the meta-data
        pooled_encounter_entities = self.encounter_pooler(
            extracted_entities["encounter_entities"], day_encounter_coupling_mask
        )
        pooled_encounter_entities = self.post_pooling_mlp(pooled_encounter_entities)
        meta_data = torch.cat(
            [embeddings["embedded_history_days"], pooled_encounter_entities], dim=-1
        )
        entities = torch.cat(
            [extracted_entities["daily_entities"], pooled_encounter_entities], dim=-1
        )
        for sab in self.attention_blocks:
            entities = sab(entities)
            entities = self.entity_masker(entities, inputs["valid_history_mask"])
            entities = torch.cat([entities, meta_data], dim=-1)
        # Now we process the entities with the two MLPs, one for infectiousness
        # and the other for contagion prediction
        encounter_variables = self.encounter_mlp(entities)
        latent_variable = self.latent_variable_mlp(entities)
        # Done: pack and return
        assert (
            not self._diagnose or not self._output_as_tuple
        ), "cannot produce tuple (for tracing) while diagnosing"
        if self._output_as_tuple:
            return encounter_variables, latent_variable
        results = dict()
        results["encounter_variables"] = encounter_variables
        results["latent_variable"] = latent_variable
        if self._diagnose:
            _locals = dict(locals())
            _locals.pop("results")
            _locals.pop("self")
            _locals.pop("encounter_variables")
            _locals.pop("latent_variable")
            results.update(_locals)
        return results


class DiurnalContactTracingTransformerV2(_DiurnalContactTracingTransformerV2):
    def __init__(
        self,
        *,
        # Embeddings
        capacity=128,
        dropout=0.1,
        num_health_history_features=28,
        health_history_embedding_dim=64,
        num_health_profile_features=13,
        health_profile_embedding_dim=32,
        use_learned_time_embedding=True,
        time_embedding_dim=32,
        encounter_duration_embedding_dim=32,
        encounter_duration_embedding_mode="sines",
        encounter_duration_thermo_range=(0.0, 6.0),
        encounter_duration_num_thermo_bins=32,
        num_encounter_partner_id_bits=16,
        use_encounter_partner_id_embedding=False,
        encounter_partner_id_embedding_dim=32,
        message_dim=1,
        message_embedding_dim=128,
        # Attention
        pooled_encounter_dim=64,
        num_heads=4,
        attention_block_capacity=128,
        num_attention_blocks=2,
        # Output
        encounter_output_features=1,
        latent_variable_output_features=1,
    ):
        # ------- Embeddings -------
        health_history_embedding = mods.HealthHistoryEmbedding(
            in_features=num_health_history_features,
            embedding_size=health_history_embedding_dim,
            capacity=capacity,
            dropout=dropout,
        )
        health_profile_embedding = mods.HealthProfileEmbedding(
            in_features=num_health_profile_features,
            embedding_size=health_profile_embedding_dim,
            capacity=capacity,
            dropout=dropout,
        )
        if use_learned_time_embedding:
            time_embedding = mods.TimeEmbedding(embedding_size=time_embedding_dim)
        else:
            time_embedding = mods.PositionalEncoding(encoding_dim=time_embedding_dim)
        if encounter_duration_embedding_mode == "thermo":
            duration_embedding = mods.DurationEmbedding(
                num_thermo_bins=encounter_duration_num_thermo_bins,
                embedding_size=encounter_duration_embedding_dim,
                thermo_range=encounter_duration_thermo_range,
                capacity=capacity,
                dropout=dropout,
            )
        elif encounter_duration_embedding_mode == "sines":
            duration_embedding = mods.PositionalEncoding(
                encoding_dim=encounter_duration_embedding_dim
            )
        else:
            raise ValueError
        if use_encounter_partner_id_embedding:
            partner_id_embedding = mods.PartnerIdEmbedding(
                num_id_bits=num_encounter_partner_id_bits,
                embedding_size=encounter_partner_id_embedding_dim,
            )
        else:
            partner_id_embedding = None
        message_embedding = mods.MessageEmbedding(
            message_dim=message_dim,
            embedding_size=message_embedding_dim,
            capacity=capacity,
            dropout=dropout,
        )
        # ------- Attentions -------
        metadata_dim = time_embedding_dim + pooled_encounter_dim
        daily_in_dim = (
            time_embedding_dim
            + health_history_embedding_dim
            + health_profile_embedding_dim
            + pooled_encounter_dim
        )
        daily_intermediate_dim = attention_block_capacity + metadata_dim
        pooler_in_dim = (
            time_embedding_dim
            + encounter_partner_id_embedding_dim
            + encounter_duration_embedding_dim
            + health_history_embedding_dim
            + message_embedding_dim
            + health_profile_embedding_dim
        )
        # Make attention pooler
        encounter_pooler = attn.PMA(
            dim=pooler_in_dim, num_heads=num_heads, num_seeds=14
        )
        post_pooling_mlp = nn.Sequential(
            nn.Linear(pooler_in_dim, capacity),
            nn.ReLU(),
            nn.Linear(capacity, pooled_encounter_dim),
        )
        # Make the SABs
        attention_blocks = [
            attn.SAB(
                dim_in=daily_in_dim,
                dim_out=attention_block_capacity,
                num_heads=num_heads,
            )
        ] + [
            attn.SAB(
                dim_in=daily_intermediate_dim,
                dim_out=attention_block_capacity,
                num_heads=num_heads,
            )
            for _ in range(num_attention_blocks - 1)
        ]
        attention_blocks = nn.ModuleList(attention_blocks)
        # ------- Output processors -------
        # Encounter
        encounter_mlp = nn.Sequential(
            nn.Linear(attention_block_capacity + metadata_dim, capacity),
            nn.ReLU(),
            nn.Linear(capacity, encounter_output_features),
        )
        # Latent variables
        latent_variable_mlp = nn.Sequential(
            nn.Linear(attention_block_capacity + metadata_dim, capacity),
            nn.ReLU(),
            nn.Linear(capacity, latent_variable_output_features),
        )
        # ------- Placeholders -------
        partner_id_placeholder = nn.Parameter(
            torch.randn((encounter_partner_id_embedding_dim,))
        )
        # ------- Masking -------
        entity_masker = mods.EntityMasker()
        super(DiurnalContactTracingTransformerV2, self).__init__(
            health_history_embedding=health_history_embedding,
            health_profile_embedding=health_profile_embedding,
            time_embedding=time_embedding,
            duration_embedding=duration_embedding,
            partner_id_embedding=partner_id_embedding,
            message_embedding=message_embedding,
            encounter_pooler=encounter_pooler,
            post_pooling_mlp=post_pooling_mlp,
            attention_blocks=attention_blocks,
            latent_variable_mlp=latent_variable_mlp,
            encounter_mlp=encounter_mlp,
            entity_masker=entity_masker,
            partner_id_placeholder=partner_id_placeholder,
        )
