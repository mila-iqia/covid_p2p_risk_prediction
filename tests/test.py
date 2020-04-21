import unittest 
import sys
sys.path.append('..')

class TestDataLoader(unittest.TestCase):
    def test(self):
        from loader import get_dataloader
        path = "../output.pkl"
        batch_size=5
        dataloader = get_dataloader(batch_size=batch_size, shuffle=False, num_workers=2, path=path)
        batch = next(iter(dataloader))
        self.assertEqual(len(batch),10)
        #Testing that all the keys in the batch have the batch_size
        keys_in_batch = list(batch.keys())
        for key in keys_in_batch:
            self.assertEqual(len(batch[key]),batch_size)


class TestModel(unittest.TestCase):
    def test(self):
        from loader import ContactDataset
        from torch.utils.data import DataLoader
        from models import ContactTracingTransformer
        batch_size=5
        path = "../output.pkl"
        dataset = ContactDataset(path)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=ContactDataset.collate_fn)
        batch = next(iter(dataloader))

        ctt = ContactTracingTransformer(pool_latent_entities=False, use_logit_sink=False)
        output = ctt(batch)
        # print(output.latent_variable.shape)
        self.assertEqual(output.latent_variable.shape[0],batch_size)
        self.assertEqual(output.encounter_variables.shape[0],batch_size)
        # print(output.encounter_variables.shape)

if __name__ == '__main__': 
    unittest.main() 