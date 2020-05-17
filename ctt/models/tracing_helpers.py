import torch.jit


@torch.jit.script
def get_embedding_meta_data(
        entities: torch.Tensor,
        embedded_history_days: torch.Tensor,
        embedded_encounter_partner_ids: torch.Tensor,
        embedded_encounter_duration: torch.Tensor,
) -> torch.Tensor:
    meta_data_dim = (
            embedded_history_days.shape[2]
            + embedded_encounter_partner_ids.shape[2]
            + embedded_encounter_duration.shape[2]
    )
    return entities[:, :, :meta_data_dim]


@torch.jit.script
def get_pre_encounter_variables(
        entities: torch.Tensor,
        embedded_history_days: torch.Tensor,
        embedded_encounter_partner_ids: torch.Tensor,
        embedded_encounter_duration: torch.Tensor,
        num_encounters: torch.Tensor,
) -> torch.Tensor:
    meta_data_dim = (
            embedded_history_days.shape[2]
            + embedded_encounter_partner_ids.shape[2]
            + embedded_encounter_duration.shape[2]
    )
    return entities[:, :num_encounters, meta_data_dim:]


@torch.jit.script
def get_pre_latent_variable(
        entities: torch.Tensor,
        num_encounters: torch.Tensor,
) -> torch.Tensor:
    return entities[:, num_encounters:]


@torch.jit.script
def get_embedded_encounter_partner_ids(
        partner_id_placeholder: torch.Tensor,
        num_encounters: torch.Tensor,
        batch_size: torch.Tensor,
) -> torch.Tensor:
    return partner_id_placeholder[
        None, None
    ].expand(
        batch_size.item(),
        num_encounters.item(),
        partner_id_placeholder.shape[-1]
    )
