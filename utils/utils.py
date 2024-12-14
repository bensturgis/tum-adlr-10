import torch
from torch.utils.data import TensorDataset

def combine_datasets(dataset1, dataset2):
    """
    Combines two TensorDatasets by concatenating their tensors along the first dimension.
    """
    tensors1 = dataset1.tensors
    tensors2 = dataset2.tensors
    combined_tensors = [torch.cat([t1, t2], dim=0) for t1, t2 in zip(tensors1, tensors2)]
    
    return TensorDataset(*combined_tensors)
