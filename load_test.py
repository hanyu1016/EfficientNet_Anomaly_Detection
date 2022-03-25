import torch
from models.efficientNet import MyEfficientNet

net = MyEfficientNet()
net.load_state_dict(torch.load('model/test_C+R_haveword_data_baseline/net_020.pth'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    net = nn.DataParallel(net)
net.to(device)
net = net.to(device)