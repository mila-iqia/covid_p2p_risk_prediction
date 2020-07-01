from typing import Union

import torch
import torch.nn as nn

from ctt.models.transformers.ctt0 import _ContactTracingTransformer
import ctt.models.attn as attn
import ctt.models.modules as mods
import ctt.utils as cu


class _MixSetNet(_ContactTracingTransformer):
    def _attention_loop(
        self,
        entities: torch.Tensor,
        meta_data: torch.Tensor,
        attention_mask: torch.Tensor,
        expanded_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Let'er rip!
        # noinspection PyTypeChecker
        for sab in self.self_attention_blocks:
            if isinstance(sab, attn.SRB):
                mask = expanded_mask
            elif isinstance(sab, attn.SAB):
                mask = attention_mask
            else:
                raise TypeError
            entities = sab(entities, weights=mask)
            entities = self.entity_masker(entities, expanded_mask)
            # Append meta-data for the next round of message passing
            entities = torch.cat([meta_data, entities], dim=2)
        return entities


class MixSetNet(_MixSetNet):
    SRB_BLOCK_TYPE = "r"
    SAB_BLOCK_TYPE = "s"

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
        message_embedding_mode="mlp",
        # Attention
        num_heads=4,
        block_capacity=128,
        srb_aggregation="max",
        srb_feature_size_divisor=1,
        block_types=f"{SRB_BLOCK_TYPE}{SAB_BLOCK_TYPE}",
        use_layernorm=False,
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
        if message_embedding_mode == "mlp":
            message_embedding = mods.MessageEmbedding(
                message_dim=message_dim,
                embedding_size=message_embedding_dim,
                capacity=capacity,
                dropout=dropout,
            )
        elif message_embedding_mode == "sines":
            assert message_dim == 1
            message_embedding = mods.PositionalEncoding(
                encoding_dim=message_embedding_dim,
            )
        else:
            raise NotImplementedError
        # ------- Attention -------
        block_in_dim = (
            time_embedding_dim
            + encounter_partner_id_embedding_dim
            + encounter_duration_embedding_dim
            + health_history_embedding_dim
            + message_embedding_dim
            + health_profile_embedding_dim
        )
        block_metadata_dim = (
            time_embedding_dim
            + encounter_partner_id_embedding_dim
            + encounter_duration_embedding_dim
        )
        block_intermediate_in_dim = block_capacity + block_metadata_dim
        blocks = []
        # Build the blocks
        for block_idx, block_type in enumerate(block_types):
            dim_in = block_in_dim if block_idx == 0 else block_intermediate_in_dim
            if block_type == self.SRB_BLOCK_TYPE:
                blocks.append(
                    attn.SRB(
                        dim_in=dim_in,
                        dim_out=block_capacity,
                        feature_size=block_capacity // srb_feature_size_divisor,
                        aggregation=srb_aggregation,
                    )
                )
            elif block_type == self.SAB_BLOCK_TYPE:
                blocks.append(
                    attn.SAB(
                        dim_in=dim_in,
                        dim_out=block_capacity,
                        num_heads=num_heads,
                        ln=use_layernorm,
                    )
                )
        blocks = nn.ModuleList(blocks)
        # ------- Output processors -------
        if latent_variable_output_features == "num_bins":
            latent_variable_output_features = len(cu.get_infectiousness_bins()) + 1
        # Encounter
        encounter_mlp = nn.Sequential(
            nn.Linear(block_capacity, capacity),
            nn.ReLU(),
            nn.Linear(capacity, encounter_output_features),
        )
        # Latent variables
        if isinstance(latent_variable_output_features, int):
            latent_variable_mlp = nn.Sequential(
                nn.Linear(block_capacity + block_metadata_dim, capacity),
                nn.ReLU(),
                nn.Linear(capacity, latent_variable_output_features),
            )
        elif isinstance(latent_variable_output_features, dict):
            latent_variable_mlp = {}
            for key in latent_variable_output_features:
                latent_variable_mlp[key] = nn.Sequential(
                    nn.Linear(block_capacity + block_metadata_dim, capacity),
                    nn.ReLU(),
                    nn.Linear(capacity, latent_variable_output_features[key]),
                )
            latent_variable_mlp = nn.ModuleDict(latent_variable_mlp)
        else:
            raise TypeError
        # ------- Output placeholders -------
        # noinspection PyArgumentList
        message_placeholder = nn.Parameter(torch.randn((message_embedding_dim,)))
        # noinspection PyArgumentList
        partner_id_placeholder = nn.Parameter(
            torch.randn((encounter_partner_id_embedding_dim,))
        )
        # noinspection PyArgumentList
        duration_placeholder = nn.Parameter(
            torch.randn((encounter_duration_embedding_dim,))
        )
        # ------- Masking -------
        entity_masker = mods.EntityMasker()
        # Done; init the super
        super(MixSetNet, self).__init__(
            health_history_embedding=health_history_embedding,
            health_profile_embedding=health_profile_embedding,
            time_embedding=time_embedding,
            duration_embedding=duration_embedding,
            partner_id_embedding=partner_id_embedding,
            message_embedding=message_embedding,
            self_attention_blocks=blocks,
            latent_variable_mlp=latent_variable_mlp,
            encounter_mlp=encounter_mlp,
            entity_masker=entity_masker,
            message_placeholder=message_placeholder,
            partner_id_placeholder=partner_id_placeholder,
            duration_placeholder=duration_placeholder,
        )
