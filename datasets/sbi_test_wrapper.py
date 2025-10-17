import torch

class SBITestWrapper:
    def __init__(self, dataset):
        self.dataset = dataset
        self.dataset.split = "test"
        self.dataset.train = True
        self.dataset.return_vid_id = True


    def __getattr__(self, i):
        return getattr(self.dataset, i)
    
    def __len__(self):
        return 2*len(self.dataset)
    
    def __getitem__(self, idx):
        i, f = idx//2, idx%2

        img_f, _, _, _, img_r, _, _, _, id = self.dataset[i]
        id = f"{id}_{f}"

        return (img_f if f else img_r), torch.tensor([f]), id