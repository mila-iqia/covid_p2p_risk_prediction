import numpy as np
import unittest

import ctt.inference.infer
import ctt.data_loading
import covid19infserver.server_utils


class Tests(unittest.TestCase):

    DATASET_PATH = "../data/1k-1-output"
    EXPERIMENT_PATH = "../exp/DEBUG-0"
    NUM_KEYS_IN_BATCH = 15

    def test_inference(self):
        manager = covid19infserver.server_utils.InferenceBroker(
            model_exp_path=self.EXPERIMENT_PATH,
            workers=2,
            mp_backend="loky",
            mp_threads=4,
            port=6688,
            verbose=False,
        )
        manager.start()
        local_engine = ctt.inference.infer.InferenceEngine(self.EXPERIMENT_PATH)
        remote_engine = covid19infserver.server_utils.InferenceClient(6688, "localhost")
        dataset = ctt.data_loading.loader.ContactDataset(self.DATASET_PATH)
        for _ in range(1000):
            hdi, local_output = None, None
            try:
                hdi = dataset.read(
                    np.random.randint(dataset.num_humans),
                    np.random.randint(dataset.num_days),
                )
                local_output = local_engine.infer(hdi)
            except ctt.data_loading.loader.InvalidSetSize:
                pass  # skip samples without encounters
            # avoid sending a bad sample to remote server
            remote_output = remote_engine.infer(hdi)
            if local_output is None:
                self.assertTrue(remote_output is None)
            # TODO: test below is pretty useless until we figure out the output
            #if local_output is not None:
            if local_output is not None and remote_output is not None:
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
                # self.assertTrue(
                #     np.isclose(
                #         local_output["contagion_proba"],
                #         remote_output["contagion_proba"],
                #     ).all()
                # )
                # self.assertTrue(
                #     np.isclose(
                #         local_output["infectiousness"], remote_output["infectiousness"]
                #     ).all()
                # )
        manager.stop()
        manager.join()


if __name__ == "__main__":
    unittest.main()
