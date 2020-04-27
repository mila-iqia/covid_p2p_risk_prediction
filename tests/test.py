import unittest


class Tests(unittest.TestCase):

    DATASET_PATH = "../data/1k-1-output"
    ZIP_DATASET_PATH = "../data/1k-1-output.zip"
    NUM_KEYS_IN_BATCH = 12

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

        ctt = ContactTracingTransformer(
            pool_latent_entities=False, use_logit_sink=False
        )
        output = Dict(ctt(batch))
        # print(output.latent_variable.shape)
        self.assertEqual(output.latent_variable.shape[0], batch_size)
        self.assertEqual(output.encounter_variables.shape[0], batch_size)
        # print(output.encounter_variables.shape)

    def test_losses(self):
        from loader import ContactDataset
        from torch.utils.data import DataLoader
        from models import ContactTracingTransformer
        from losses import ContagionLoss
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
            self.assertEqual(
                dataset.extract(sample, "preexisting_conditions").shape[-1],
                len(dataset.DEFAULT_PREEXISTING_CONDITIONS),
            )
            self.assertEqual(dataset.extract(sample, "test_results").shape[-1], 1)
            self.assertEqual(dataset.extract(sample, "age").shape[-1], 1)
            self.assertEqual(dataset.extract(sample, "sex").shape[-1], 1)
            self.assertEqual(
                dataset.extract(sample, "reported_symptoms_at_encounter").shape[-1], 12
            )
            self.assertEqual(
                dataset.extract(sample, "test_results_at_encounter").shape[-1], 1
            )

        sample = dataset.get(890, 25)
        self.assertIsInstance(sample, Dict)
        # validate(sample)

        if self.ZIP_DATASET_PATH is not None:
            dataset = ContactDataset(self.ZIP_DATASET_PATH, preload=False)
            sample = dataset.get(890, 25)
            self.assertIsInstance(sample, Dict)


    def test_tflite_model_conversion(self):
        from models import ContactTracingTransformer
        from export_to_tflite import convert_pytorch_model_fixed_messages

        # Instantiate new model
        model = ContactTracingTransformer()
        model.eval()

        # Test the conversion to TFLite
        for nb_messages in [10, 50, 100]:
            max_diff = convert_pytorch_model_fixed_messages(
                model, nb_messages, working_directory="./tmp/test_dir/")
            self.assertLess(max_diff, 0.005)


if __name__ == "__main__":
    unittest.main()
