import unittest


class Tests(unittest.TestCase):

    DATASET_PATH = ZIP_PATH = (
        "../data/sim_v2_people-1000_days-30_init-0.003570583805768116_uptake-"
        "0.9814107861305532_seed-3098_20200629-230107_414875.zip"
    )
    DATASET_DIR_PATH = "../data/payload"
    EXP_DIR = "/Users/nrahaman/Python/ctt/tmp/CTT-SHIPMENT-0"
    NUM_KEYS_IN_BATCH = 15

    def test_model_runs(self):
        from ctt.data_loading.loader import ContactDataset
        from torch.utils.data import DataLoader
        from ctt.models.transformer import (
            ContactTracingTransformer,
            DiurnalContactTracingTransformer,
            DiurnalContactTracingTransformerV2,
            MixSetNet,
        )
        from ctt.models.moment_processors import MomentNet
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

        ctt = DiurnalContactTracingTransformer()
        test_output(ctt)

        ctt = DiurnalContactTracingTransformerV2()
        test_output(ctt)

        ctt = DiurnalContactTracingTransformerV2()
        test_output(ctt)

        ctt = MixSetNet()
        test_output(ctt)

        ctt = MixSetNet(srb_aggregation="mean")
        test_output(ctt)

        mnet = MomentNet(block_types="l")
        test_output(mnet)

    def test_tf_compat(self):
        import torch
        from ctt.data_loading.loader import ContactDataset
        from torch.utils.data import DataLoader
        from ctt.models.transformer import ContactTracingTransformer
        from ctt.models.attn import MAB
        from addict import Dict

        batch_size = 5
        path = self.DATASET_PATH
        dataset = ContactDataset(path)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, collate_fn=ContactDataset.collate_fn
        )
        batch = next(iter(dataloader))

        ctt = ContactTracingTransformer()
        ctt.eval()

        MAB.TF_COMPAT = False
        output_no_compat = ctt(batch)

        MAB.TF_COMPAT = True
        output_tf_compat = ctt(batch)

        for key in output_no_compat:
            self.assertTrue(
                torch.allclose(output_no_compat[key], output_tf_compat[key])
            )

    def test_model_padding(self):
        import torch
        from ctt.data_loading.loader import ContactDataset
        from torch.utils.data import DataLoader
        from ctt.models.transformer import (
            ContactTracingTransformer,
            DiurnalContactTracingTransformer,
            DiurnalContactTracingTransformerV2,
            MixSetNet,
        )

        torch.random.manual_seed(43)

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

        def _test_model(model, test_latents_only=False, atol=1e-7):
            with torch.no_grad(), model.diagnose():
                output = model(batch)
                padded_output = model(padded_batch)

            encounter_soll_wert = output["encounter_variables"][..., 0]
            encounter_ist_wert = padded_output["encounter_variables"][
                ..., :-pad_size, 0
            ]
            latent_soll_wert = output["latent_variable"]
            latent_ist_wert = padded_output["latent_variable"]
            if not test_latents_only:
                self.assertSequenceEqual(
                    encounter_soll_wert.shape, encounter_ist_wert.shape
                )
            self.assertSequenceEqual(latent_ist_wert.shape, latent_soll_wert.shape)
            if not test_latents_only:
                self.assertTrue(
                    torch.allclose(encounter_soll_wert, encounter_ist_wert, atol=atol)
                )
            self.assertTrue(
                torch.allclose(latent_soll_wert, latent_ist_wert, atol=atol)
            )

        # noinspection PyUnresolvedReferences
        ctt = ContactTracingTransformer(num_sabs=1).eval()
        _test_model(ctt)

        # noinspection PyUnresolvedReferences
        ctt = DiurnalContactTracingTransformer().eval()
        _test_model(ctt, test_latents_only=True)

        ctt = DiurnalContactTracingTransformerV2().eval()
        # FIXME Understand why we need atol this large.
        _test_model(ctt, test_latents_only=True, atol=1e-4)

        ctt = MixSetNet().eval()
        _test_model(ctt)

    def test_model_jit(self):
        import torch
        from ctt.data_loading.loader import ContactDataset
        from torch.utils.data import DataLoader
        from ctt.models.transformer import ContactTracingTransformer
        from addict import Dict

        batch_size = 5
        path = self.DATASET_PATH
        dataset = ContactDataset(path)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=ContactDataset.collate_fn,
            shuffle=False,
        )
        dataloader_iter = iter(dataloader)
        batch = next(dataloader_iter)
        model: ContactTracingTransformer = ContactTracingTransformer()
        model.eval()
        with model.output_as_tuple():
            trace = torch.jit.trace(model, (batch,),)
        # Compare trace outputs with another batch
        batch = next(dataloader_iter)
        model_output = model(batch)
        trace_output = ContactTracingTransformer.output_tuple_to_dict(trace(batch))
        for key in model_output.keys():
            self.assertTrue(torch.allclose(model_output[key], trace_output[key]))

    def test_losses(self):
        from ctt.data_loading.loader import ContactDataset
        from torch.utils.data import DataLoader
        from ctt.models.transformer import (
            ContactTracingTransformer,
            DiurnalContactTracingTransformer,
        )
        from ctt.losses import ContagionLoss, InfectiousnessLoss
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

        ctt = DiurnalContactTracingTransformer()
        output = Dict(ctt(batch))

        loss_fn = ContagionLoss(allow_multiple_exposures=True, diurnal_exposures=True)
        loss = loss_fn(batch, output)

        loss_fn = InfectiousnessLoss()
        loss = loss_fn(batch, output)

        with self.assertRaises(Exception):
            loss_fn = ContagionLoss(allow_multiple_exposures=True)
            loss = loss_fn(batch, output)

    def test_loader(self):
        from ctt.data_loading.loader import get_dataloader

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

        dataloader = get_dataloader(
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            path=self.DATASET_DIR_PATH,
        )
        batch = next(iter(dataloader))

    def test_loader_with_multiprocessing(self):
        from ctt.data_loading.loader import get_dataloader

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
        from ctt.data_loading.loader import ContactDataset, InvalidSetSize
        from addict import Dict

        path = self.DATASET_PATH

        def validate(sample):
            self.assertIsInstance(sample, Dict)
            self.assertEqual(
                dataset.extract(dataset, sample, "preexisting_conditions").shape[-1],
                len(dataset.DEFAULT_PREEXISTING_CONDITIONS),
            )
            self.assertEqual(
                dataset.extract(dataset, sample, "test_results").shape[-1], 1
            )
            self.assertEqual(dataset.extract(dataset, sample, "age").shape[-1], 1)
            self.assertEqual(dataset.extract(dataset, sample, "sex").shape[-1], 1)
            self.assertEqual(
                dataset.extract(
                    dataset, sample, "reported_symptoms_at_encounter"
                ).shape[-1],
                27,
            )
            self.assertEqual(
                dataset.extract(dataset, sample, "test_results_at_encounter").shape[-1],
                1,
            )

        dataset = ContactDataset(path)
        sample = dataset.get(67, 13, 1)
        validate(sample)

        dataset = ContactDataset(path, bit_encoded_messages=False)
        sample = dataset.get(890, 3, 1)
        validate(sample)

        dataset = ContactDataset(path, bit_encoded_messages=False)
        sample = dataset[0]
        validate(sample)

    def test_tflite_model_conversion(self):
        from ctt.models.transformer import ContactTracingTransformer
        from ctt.conversion.export_to_tflite import convert_pytorch_model_fixed_messages

        # Instantiate new model
        model = ContactTracingTransformer()
        model.eval()

        # Test the conversion to TFLite
        for nb_messages in [10, 50, 100]:
            max_diff = convert_pytorch_model_fixed_messages(
                model,
                nb_messages,
                working_directory="./tmp/test_dir/",
                dataset_path=self.DATASET_PATH,
            )
            self.assertLess(max_diff, 0.005)

    def test_inference_engine_determinism(self):
        from ctt.data_loading.loader import ContactDataset
        from ctt.inference.infer import InferenceEngine
        import numpy as np

        path = self.DATASET_PATH
        dataset = ContactDataset(path)
        num_idxs = 10
        for idx in range(num_idxs):
            filename = dataset._files[idx]
            hdi = dataset.read(file_name=filename)
            engine = InferenceEngine(self.EXP_DIR)
            output_1 = engine.infer(hdi)
            output_2 = engine.infer(hdi)
            for key in output_1:
                self.assert_(np.all(output_1[key] == output_2[key]))


if __name__ == "__main__":
    unittest.main()
