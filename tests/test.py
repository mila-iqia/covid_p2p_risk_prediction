import unittest


class Tests(unittest.TestCase):

    DATASET_PATH = "../data/1k-1-output"
    ZIP_PATH = (
        "../data/covi-1k-04-27/"
        "sim_v2_people-1000_days-60_init-0.01_seed-1_20200427-100605/"
        "output.zip"
    )
    NUM_KEYS_IN_BATCH = 15

    def test_model_runs(self):
        from loader import ContactDataset
        from torch.utils.data import DataLoader
        from models import ContactTracingTransformer
        from addict import Dict

        batch_size = 5
        path = self.DATASET_PATH
        dataset = ContactDataset(path)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, collate_fn=ContactDataset.collate_fn
        )
        batch = next(iter(dataloader))

        def test_output(mod):
            output = Dict(mod(batch))
            # print(output.latent_variable.shape)
            self.assertEqual(output.latent_variable.shape[0], batch_size)
            self.assertEqual(output.encounter_variables.shape[0], batch_size)

        ctt = ContactTracingTransformer()
        test_output(ctt)

        ctt = ContactTracingTransformer(use_encounter_partner_id_embedding=False)
        test_output(ctt)

        ctt = ContactTracingTransformer(use_learned_time_embedding=True)
        test_output(ctt)

        ctt = ContactTracingTransformer(encounter_duration_embedding_mode="sines")
        test_output(ctt)

    def test_model_padding(self):
        import torch
        from loader import ContactDataset
        from torch.utils.data import DataLoader
        from models import ContactTracingTransformer

        torch.random.manual_seed(42)

        batch_size = 5
        path = self.DATASET_PATH
        dataset = ContactDataset(path)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, collate_fn=ContactDataset.collate_fn
        )
        batch = next(iter(dataloader))

        # Padding test -- pad everything that has to do with encounters, and check
        # whether it changes results
        pad_size = 1

        def pad(tensor):
            if tensor.dim() == 3:
                zeros = torch.zeros(
                    (tensor.shape[0], pad_size, tensor.shape[2]), dtype=tensor.dtype
                )
            else:
                zeros = torch.zeros((tensor.shape[0], pad_size), dtype=tensor.dtype)
            return torch.cat([tensor, zeros], dim=1)

        padded_batch = {
            key: (pad(tensor) if key.startswith("encounter") else tensor)
            for key, tensor in batch.items()
        }
        # Pad the mask
        padded_batch["mask"] = pad(padded_batch["mask"])

        # Make the model and set it to eval
        # noinspection PyUnresolvedReferences
        ctt = ContactTracingTransformer(num_sabs=1).eval()
        with torch.no_grad(), ctt.diagnose():
            output = ctt(batch)
            padded_output = ctt(padded_batch)

        encounter_soll_wert = output["encounter_variables"][..., 0]
        encounter_ist_wert = padded_output["encounter_variables"][..., :-pad_size, 0]
        latent_soll_wert = output["latent_variable"]
        latent_ist_wert = padded_output["latent_variable"]
        self.assertSequenceEqual(encounter_soll_wert.shape, encounter_ist_wert.shape)
        self.assertSequenceEqual(latent_ist_wert.shape, latent_soll_wert.shape)
        self.assert_(torch.allclose(encounter_soll_wert, encounter_ist_wert))
        self.assert_(torch.allclose(latent_soll_wert, latent_ist_wert))

    def test_losses(self):
        from loader import ContactDataset
        from torch.utils.data import DataLoader
        from models import ContactTracingTransformer
        from losses import ContagionLoss, InfectiousnessLoss
        from addict import Dict

        batch_size = 5
        path = self.DATASET_PATH
        dataset = ContactDataset(path)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, collate_fn=ContactDataset.collate_fn
        )
        batch = next(iter(dataloader))

        ctt = ContactTracingTransformer()
        output = Dict(ctt(batch))

        loss_fn = ContagionLoss(allow_multiple_exposures=True)
        loss = loss_fn(batch, output)
        loss_fn = ContagionLoss(allow_multiple_exposures=False)
        loss = loss_fn(batch, output)
        loss_fn = InfectiousnessLoss()
        loss = loss_fn(batch, output)

    def test_loader(self):
        from loader import get_dataloader

        path = self.DATASET_PATH
        batch_size = 5
        dataloader = get_dataloader(
            batch_size=batch_size, shuffle=False, num_workers=0, path=path
        )
        batch = next(iter(dataloader))
        self.assertEqual(len(batch), self.NUM_KEYS_IN_BATCH)
        # Testing that all the keys in the batch have the batch_size
        keys_in_batch = list(batch.keys())
        for key in keys_in_batch:
            self.assertEqual(len(batch[key]), batch_size)
        dataloader = get_dataloader(
            batch_size=batch_size, shuffle=False, num_workers=0, path=[path, path]
        )
        batch = next(iter(dataloader))

    def test_loader_with_multiprocessing(self):
        from loader import get_dataloader

        path = self.DATASET_PATH
        batch_size = 5
        dataloader = get_dataloader(
            batch_size=batch_size, shuffle=False, num_workers=2, path=path
        )
        batch = next(iter(dataloader))
        self.assertEqual(len(batch), self.NUM_KEYS_IN_BATCH)
        # Testing that all the keys in the batch have the batch_size
        keys_in_batch = list(batch.keys())
        for key in keys_in_batch:
            self.assertEqual(len(batch[key]), batch_size)

    def test_dataset(self):
        from loader import ContactDataset
        from addict import Dict

        path = self.DATASET_PATH
        dataset = ContactDataset(path)

        def validate(sample):
            self.assertIsInstance(sample, Dict)
            self.assertEqual(
                dataset.extract(sample, "preexisting_conditions").shape[-1],
                len(dataset.DEFAULT_PREEXISTING_CONDITIONS),
            )
            self.assertEqual(dataset.extract(sample, "test_results").shape[-1], 1)
            self.assertEqual(dataset.extract(sample, "age").shape[-1], 8)
            self.assertEqual(dataset.extract(sample, "sex").shape[-1], 1)
            self.assertEqual(
                dataset.extract(sample, "reported_symptoms_at_encounter").shape[-1], 12
            )
            self.assertEqual(
                dataset.extract(sample, "test_results_at_encounter").shape[-1], 1
            )

        sample = dataset.get(890, 5)
        validate(sample)

        dataset = ContactDataset(self.ZIP_PATH)
        sample = dataset.get(420, 40)
        pass


if __name__ == "__main__":
    unittest.main()
