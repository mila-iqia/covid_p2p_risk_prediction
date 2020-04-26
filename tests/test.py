import unittest


class Tests(unittest.TestCase):

    DATASET_PATH = "../data/1k-1-output"
    EXPERIMENT_PATH = "../exp/DEBUG-0"
    NUM_KEYS_IN_BATCH = 13

    def test_model(self):
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

        ctt = ContactTracingTransformer(
            pool_latent_entities=False, use_logit_sink=False
        )
        test_output(ctt)

        ctt = ContactTracingTransformer(
            pool_latent_entities=False,
            use_logit_sink=False,
            use_encounter_partner_id_embedding=False,
        )
        test_output(ctt)

        ctt = ContactTracingTransformer(
            pool_latent_entities=False,
            use_logit_sink=False,
            use_learned_time_embedding=True,
        )
        test_output(ctt)

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

        ctt = ContactTracingTransformer(
            pool_latent_entities=False, use_logit_sink=False
        )
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
        sample = dataset.get(890, 5)
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

    def test_inference(self):
        import numpy as np
        import infer
        import loader

        # this should automatically load-balance inference across servers; see:
        # https://learning-0mq-with-pyzmq.readthedocs.io/en/latest/pyzmq/multiprocess/multiprocess.html
        test_ports = [6688, 6689, 6690]
        servers = [
            infer.InferenceServer(self.EXPERIMENT_PATH, port) for port in test_ports
        ]
        _ = [s.start() for s in servers]
        local_engine = infer.InferenceEngine(self.EXPERIMENT_PATH)
        remote_engine = infer.InferenceClient(test_ports, "localhost")
        dataset = loader.ContactDataset(self.DATASET_PATH)
        for _ in range(1000):
            hdi, local_output = None, None
            try:
                hdi = dataset.read(
                    np.random.randint(dataset.num_humans),
                    np.random.randint(dataset.num_days),
                )
                local_output = local_engine.infer(hdi)
            except loader.InvalidSetSize:
                pass  # skip samples without encounters
            # avoid sending a bad sample to remote server
            remote_output = remote_engine.infer(hdi)
            if local_output is None:
                self.assertTrue(remote_output is None)
            if local_output is not None:
                for output in [local_output, remote_output]:
                    self.assertIsInstance(output, dict)
                    self.assertEqual(len(output), 2)
                    self.assertIn("contagion_proba", output)
                    self.assertIn("infectiousness", output)
                    self.assertIsInstance(output["contagion_proba"], np.ndarray)
                    self.assertIsInstance(output["infectiousness"], np.ndarray)
                self.assertEqual(
                    local_output["contagion_proba"].shape,
                    remote_output["contagion_proba"].shape,
                )
                self.assertEqual(
                    local_output["infectiousness"].shape,
                    remote_output["infectiousness"].shape,
                )
                self.assertTrue(
                    np.isclose(
                        local_output["contagion_proba"],
                        remote_output["contagion_proba"],
                    ).all()
                )
                self.assertTrue(
                    np.isclose(
                        local_output["infectiousness"], remote_output["infectiousness"]
                    ).all()
                )
        for s in servers:
            s.stop()
            s.join()


if __name__ == "__main__":
    unittest.main()
