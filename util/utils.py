from torch.utils.data.dataloader import default_collate
import torch
def my_collate_fn(batch):
    '''
    batch中每個元素(data, label)
    '''
    # 過濾為 None 的数據
    batch = list(filter(lambda x: x[0] is not None, batch))
    if len(batch) == 0: return torch.Tensor()
    return default_collate(batch)  # 用默认方式拼接过滤后的batch数据