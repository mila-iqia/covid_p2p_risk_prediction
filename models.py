from addict import Dict

import torch
import torch.nn as nn

import modules as mods
import attn

# Legend
#   B: batch (i.e. humans)
#   T: index of the day
#   M: message/encounter index
#   C: channel


class _ContactTracingTransformer(nn.Module):
    def __init__(
        self,
        *,
        health_embedding: nn.Module,
        time_embedding: nn.Module,
        partner_id_embedding: nn.Module,
        message_embedding: nn.Module,
        self_attention_blocks: nn.ModuleList,
        self_latent_variable_pooler: nn.Module,
        latent_variable_mlp: nn.Module,
        encounter_logit_sink_pooler: nn.Module,
        logit_sink_mlp: nn.Module,
        encounter_mlp: nn.Module,
        entity_masker: nn.Module,
        message_placeholder: nn.Parameter,
        partner_id_placeholder: nn.Parameter,
    ):
        super(_ContactTracingTransformer, self).__init__()
        self.health_embedding = health_embedding
        self.time_embedding = time_embedding
        self.partner_id_embedding = partner_id_embedding
        self.message_embedding = message_embedding
        self.self_attention_blocks = self_attention_blocks
        self.self_latent_variable_pooler = self_latent_variable_pooler
        self.latent_variable_mlp = latent_variable_mlp
        self.encounter_logit_sink_pooler = encounter_logit_sink_pooler
        self.logit_sink_mlp = logit_sink_mlp
        self.encounter_mlp = encounter_mlp
        self.entity_masker = entity_masker
        self.message_placeholder = message_placeholder
        self.partner_id_placeholder = partner_id_placeholder

    def forward(self, inputs):
        """
        Parameters
        ----------
        inputs : Dict
            An addict with the following fields:
                -> `health_history`: a B(T=14)C tensor of the 14-day health
                    history (symptoms + test results + day) of the individual.
                -> `history_days`: a B(T=14)1 tensor of the day corresponding to the
                    T dimension in `health_history`.
                -> `encounter_health`: a BMC tensor of health during an
                    encounter indexed by M.
                -> `encounter_message`: a BMC tensor of the received
                    message from the encounter partner.
                -> `encounter_day`: a BM1 tensor of the encounter day.
                -> `encounter_partner_id`: a binary  BMC tensor specifying
                    the ID of the encounter partner.
                -> `mask`: a BM1 mask tensor distinguishing the valid entries (1)
                    from padding (0) in the set valued inputs.
        Returns
        -------
        Dict
        """
        # -------- Shape Wrangling --------
        batch_size = inputs.health_history.shape[0]
        num_history_days = inputs.health_history.shape[1]
        num_encounters = inputs.encounter_health.shape[1]
        # -------- Embeddings --------
        # Embed health
        embedded_health_history = self.health_embedding(inputs.health_history)
        embedded_encounter_health = self.health_embedding(
            inputs.encounter_health, inputs.mask
        )
        # Embed time (days)
        embedded_history_days = self.time_embedding(inputs.history_days)
        embedded_encounter_day = self.time_embedding(inputs.encounter_day, inputs.mask)
        # Embed partner-IDs
        embedded_encounter_partner_ids = self.partner_id_embedding(
            inputs.encounter_partner_id, inputs.mask
        )
        # Embed messages
        embedded_encounter_messages = self.message_embedding(
            inputs.encounter_message, inputs.mask
        )
        # -------- Self Attention --------
        # Prepare the entities -- one set for the encounters and the other for self health
        encounter_entities = torch.cat(
            [
                embedded_encounter_day,
                embedded_encounter_partner_ids,
                embedded_encounter_health,
                embedded_encounter_messages,
            ],
            dim=-1,
        )
        expanded_message_placeholder = self.message_placeholder[None, None].expand(
            batch_size, num_history_days, embedded_encounter_messages.shape[-1]
        )
        expanded_pid_placeholder = self.partner_id_placeholder[None, None].expand(
            batch_size, num_history_days, embedded_encounter_partner_ids.shape[-1]
        )
        self_entities = torch.cat(
            [
                embedded_history_days,
                expanded_pid_placeholder,
                embedded_health_history,
                expanded_message_placeholder,
            ],
            dim=2,
        )
        # Concatenate encounter and self entities in to one big set (before passing to
        # the self attention blocks). In addition, expand inputs.mask to account for
        # masking the entire set of entities.
        entities = torch.cat([encounter_entities, self_entities], dim=1)
        extra_mask = torch.ones(
            size=(batch_size, num_history_days),
            dtype=inputs.mask.dtype,
            device=inputs.mask.device,
        )
        expanded_mask = torch.cat([inputs.mask, extra_mask], dim=1)
        entities = self.entity_masker(entities, expanded_mask)
        # Grab a copy of the "meta-data", which we will be appending to entities at
        # every step. These meta-data are the time-stamps and partner_ids
        meta_data_dim = (
            embedded_history_days.shape[2] + embedded_encounter_partner_ids.shape[2]
        )
        meta_data = entities[:, :, :meta_data_dim]
        # Let'er rip!
        # noinspection PyTypeChecker
        for sab in self.self_attention_blocks:
            entities = sab(entities)
            entities = self.entity_masker(entities, expanded_mask)
            # Append meta-data for the next round of message passing
            entities = torch.cat([meta_data, entities], dim=2)
        # -------- Entity Pooling --------
        # -------- Latent Variables
        # Pool the self entities together to predict one latent variable
        pre_latent_variable = self.self_latent_variable_pooler(entities)
        assert pre_latent_variable.shape[1] == 1
        pre_latent_variable = pre_latent_variable.reshape(
            batch_size, pre_latent_variable.shape[2]
        )
        # Push through the latent variable MLP to get the latent variables
        # latent_variable.shape = BC
        latent_variable = self.latent_variable_mlp(pre_latent_variable)
        # -------- Logit sink
        pre_logit_sink = self.encounter_logit_sink_pooler(entities)
        assert pre_logit_sink.shape[1] == 1
        # logit_sink.shape = B1C
        logit_sink = self.logit_sink_mlp(pre_logit_sink)
        # -------- Generate Output Variables --------
        # Process encounters to their variables
        pre_encounter_variables = torch.cat(
            [entities[:, :num_encounters, meta_data_dim:], logit_sink], dim=1
        )
        encounter_variables = self.encounter_mlp(pre_encounter_variables)
        # Done: pack to an addict and return
        results = Dict()
        results.encounter_variables = encounter_variables
        results.latent_variable = latent_variable
        return results


class ContactTracingTransformer(_ContactTracingTransformer):
    def __init__(
        self,
        *,
        # Embeddings
        capacity=128,
        dropout=0.1,
        num_health_features=13,
        health_embedding_dim=64,
        time_embedding_dim=32,
        num_encounter_partner_id_bits=16,
        encounter_partner_id_embedding_dim=128,
        message_dim=8,
        message_embedding_dim=128,
        # Attention
        num_heads=4,
        sab_capacity=128,
        num_sabs=2,
        # Output
        encounter_output_features=1,
        latent_variable_output_features=10,
    ):
        # ------- Embeddings -------
        health_embedding = mods.HealthEmbedding(
            in_features=num_health_features,
            embedding_size=health_embedding_dim,
            capacity=capacity,
            dropout=dropout,
        )
        time_embedding = mods.PositionalEncoding(encoding_dim=time_embedding_dim)
        partner_id_embedding = mods.PartnerIdEmbedding(
            num_id_bits=num_encounter_partner_id_bits,
            embedding_size=encounter_partner_id_embedding_dim,
        )
        message_embedding = mods.MessageEmbedding(
            message_dim=message_dim,
            embedding_size=message_embedding_dim,
            capacity=capacity,
            dropout=dropout,
        )
        # ------- Attention -------
        sab_in_dim = (
            time_embedding_dim
            + encounter_partner_id_embedding_dim
            + health_embedding_dim
            + message_embedding_dim
        )
        sab_metadata_dim = time_embedding_dim + encounter_partner_id_embedding_dim
        sab_intermediate_in_dim = sab_capacity + sab_metadata_dim
        # Build the SABs
        self_attention_blocks = [
            attn.SAB(dim_in=sab_in_dim, dim_out=sab_capacity, num_heads=num_heads)
        ]
        for sab_idx in range(num_sabs - 1):
            self_attention_blocks.append(
                attn.SAB(
                    dim_in=sab_intermediate_in_dim,
                    dim_out=sab_capacity,
                    num_heads=num_heads,
                )
            )
        self_attention_blocks = nn.ModuleList(self_attention_blocks)
        # Build the entity poolers
        self_latent_variable_pooler = attn.PMA(
            dim=sab_capacity + sab_metadata_dim, num_seeds=1, num_heads=num_heads
        )
        encounter_logit_sink_pooler = attn.PMA(
            dim=sab_capacity + sab_metadata_dim, num_seeds=1, num_heads=num_heads
        )
        # ------- Output processors -------
        # Encounter
        logit_sink_mlp = nn.Linear(sab_capacity + sab_metadata_dim, sab_capacity)
        encounter_mlp = nn.Sequential(
            nn.Linear(sab_capacity, capacity),
            nn.ReLU(),
            nn.Linear(capacity, encounter_output_features),
        )
        # Latent variables
        latent_variable_mlp = nn.Sequential(
            nn.Linear(sab_capacity + sab_metadata_dim, capacity),
            nn.ReLU(),
            nn.Linear(capacity, latent_variable_output_features),
        )
        # ------- Output placeholders -------
        # noinspection PyArgumentList
        message_placeholder = nn.Parameter(torch.randn((message_embedding_dim,)))
        # noinspection PyArgumentList
        partner_id_placeholder = nn.Parameter(
            torch.randn((encounter_partner_id_embedding_dim,))
        )
        # ------- Masking -------
        entity_masker = mods.EntityMasker()
        # Done; init the super
        super(ContactTracingTransformer, self).__init__(
            health_embedding=health_embedding,
            time_embedding=time_embedding,
            partner_id_embedding=partner_id_embedding,
            message_embedding=message_embedding,
            self_attention_blocks=self_attention_blocks,
            self_latent_variable_pooler=self_latent_variable_pooler,
            latent_variable_mlp=latent_variable_mlp,
            encounter_logit_sink_pooler=encounter_logit_sink_pooler,
            logit_sink_mlp=logit_sink_mlp,
            encounter_mlp=encounter_mlp,
            entity_masker=entity_masker,
            message_placeholder=message_placeholder,
            partner_id_placeholder=partner_id_placeholder,
        )


if __name__ == "__main__":
    from loader import ContactDataset
    from torch.utils.data import DataLoader

    path = "/Users/nrahaman/Python/ctt/data/output.pkl"
    dataset = ContactDataset(path)
    dataloader = DataLoader(dataset, batch_size=5, collate_fn=ContactDataset.collate_fn)
    batch = next(iter(dataloader))

    ctt = ContactTracingTransformer()
    output = ctt(batch)
