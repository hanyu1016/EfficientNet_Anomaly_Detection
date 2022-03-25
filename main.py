# from torch._C import double
from torchvision import datasets, transforms
# from loss.metrics import ArcFace, CosFace, SphereFace, Am_softmax
import torch
#import wandb
import gc
import torch.nn
import argparse
import test #Hanyu
# import testMultiThresh
import torch.optim as optim 
import torch.nn.functional as F
import argparse
import warnings
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms.functional as TF
from torch.utils.data.dataloader import default_collate
from eval_kit.detector_affect import EmotionDetector


from loss.label_smooth import LabelSmoothSoftmaxCE
from util.utils import my_collate_fn
# from loss.customFocal import FocalLoss

# from loss.focal import FocalLoss
# from dataset.customDataFerPSin import customData
from customDataJet import customData
# from dataset.customData import customData
import matplotlib.pyplot as plt
import numpy as np
import os
# from testMultiThresh import TestJet #Hanyu
from test import TestJet #Hanyu
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
warnings.filterwarnings("ignore")

from models.efficientNet import MyEfficientNet

class SquarePad():
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return TF.pad(image, padding, 0, padding_mode='edge')

def log(trainErr, trainAcc, valErr, valAcc, epoch):
    wandb.log({"Train Error": trainErr,
               "Train Accuracy": trainAcc,
               "Valid Error": valErr,
               "Valid Accuracy": valAcc},step= epoch)

def Load_test(args):
    net = MyEfficientNet()
    net.load_state_dict(torch.load(args.inference_model_path))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device).eval()

    data_transforms = {
    'test': transforms.Compose([
        SquarePad(),
        transforms.Resize((args.input_size,args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        
    ])
    }

    image_datasets = {x: customData(data_path=args.data_path, 
                                    test_data_path=args.easy_val_data_path,
                                    phase=x, 
                                    data_transforms=data_transforms,
                                    is_shuffle=False,
                                    test_path=args.test_data_path) for x in ['test']}

    # Create training and validation dataloaders

    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1, shuffle=False, num_workers=0,
                                    collate_fn=my_collate_fn) for x in ['test']}

    data_loader_target = dataloaders_dict['test']    
    amount = args.at

    with torch.no_grad():
        with open(args.inference_result_path,'w') as predict_f:
            for (img_paths, catimages, labels) in data_loader_target:
                # print(img_paths)
                #print(catimages)
                #print(labels)
                images = catimages.cuda()
                label_pred  = net(images)
                print(label_pred)
                m = torch.nn.Softmax(dim=1)
                score = m(label_pred)
                # print(score)
                score = score[:,1:]

                score = score + amount

                pred = np.round(score.cpu().numpy())
                pred = np.array(pred, dtype=int)
                pred = np.squeeze(pred, 1)

                # print(pred)
                predict_f.write(img_paths[0]+'\t'+str(pred[0])+' \n')
        predict_f.close()

def main(args):
    # Initial Model
    net = MyEfficientNet()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.is_load_check_point:  # 是否train到一半中斷
        load_epoch=0
        for i in range(args.epoch):
            file_name = '%s/net_%03d.pth' % (args.results_path, i+1)
            if os.path.isfile(file_name):
                load_epoch=i+1
            else:
                if load_epoch>0:
                    file_name = '%s/net_%03d.pth' % (args.results_path, load_epoch)
                    net.load_state_dict(torch.load(file_name))
                break
        args.pre_epoch=load_epoch
    else:
        import shutil
        print('Delete old data '+args.results_path)
        shutil.rmtree(args.results_path,ignore_errors=True)   #重train時刪除舊資料

    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net = net.to(device)

    # Data preprocessing
    data_transforms = {
        'train': transforms.Compose([
            SquarePad(),
            transforms.Resize((args.input_size,args.input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            #transforms.RandomErasing(),
        ]),

        'val': transforms.Compose([
            SquarePad(),
            transforms.Resize((args.input_size,args.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        
        'easy_val': transforms.Compose([
            SquarePad(),
            transforms.Resize((args.input_size,args.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    }

    image_datasets = {x: customData(data_path=args.data_path, 
                                    test_data_path=args.easy_val_data_path,
                                    phase=x, 
                                    data_transforms=data_transforms) for x in ['train','val', 'easy_val','test']}
    '''
    test_set_dataset = {x:customData(data_path=args.EASY_VAL_DATA_PATH,
                        #txt_path=args.txt_path,
                        phase=x, 
                        data_transforms=data_transforms) for x in ['easy_val']}
    '''

    # Create training and validation dataloaders

    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.bs, shuffle=True, num_workers=0,
                                    collate_fn=my_collate_fn) for x in ['train', 'val','easy_val','test']}
                                    
    params_to_update = list(net.parameters())

    print("Params to learn:")
    if args.feature_extract:
        params_to_update = []
        for name, param in net.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in net.named_parameters():
            if param.requires_grad == True:
                print("\t", name)
    ii = 0

    best_acc = 0  # 初始化best test accuracy
    print("Start Training, DeepNetwork!")

    # criterion: 標準準則 主要用來計算loss
    criterion = LabelSmoothSoftmaxCE()
    # criterion = FocalLoss(outNum = 8, gamma=2, weight = image_datasets['train'].class_sample_count)
    # netOutFeatureNum = net._fc.out_features
    '''
    head_dict = {'ArcFace': ArcFace(in_features = args.NET_OUT_FEATURES, out_features = 8, device_id = args.GPU_ID),
            'CosFace': CosFace(in_features = args.NET_OUT_FEATURES, out_features = 8, device_id = args.GPU_ID),
            'SphereFace': SphereFace(in_features = args.NET_OUT_FEATURES, out_features = 8, device_id = args.GPU_ID),
            'Am_softmax': Am_softmax(in_features = args.NET_OUT_FEATURES, out_features = 8, device_id = args.GPU_ID)}
    head = head_dict[args.HEAD_NAME]
    '''
    # optimizer
    optimizer = torch.optim.AdamW(params=params_to_update, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
    # scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=3, verbose=True)
    trainAllAcc = []
    trainAllLoss = []
    validAllAcc = []
    validAllLoss = []
    eps = 0
    
    if not os.path.isdir(args.results_path + "/txt"):
        os.makedirs(args.results_path + "/txt")
    if not os.path.isdir(args.results_path + "/output"):
        os.makedirs(args.results_path + "/output")

    with open(args.results_path + args.acc_txt_path, "w") as f:
        with open(args.results_path + args.log_txt_path, "w")as f2:
            for epoch in range(args.pre_epoch, args.epoch):
                gc.collect()
                torch.cuda.empty_cache()
                # scheduler.step(epoch)
                print('\nEpoch: %d' % (epoch + 1))
                net.train()
                # head.train()
                train_sum_loss = 0.0
                correct = 0.0                
                total = 0.0
                trainLossEpNow = 0.0
                trainAccEpNow = 0.0
                eps +=1

                for i, data in enumerate(dataloaders_dict['train'], 0):
                    length = len(dataloaders_dict['train'])
                    iterNow = (i + 1 + epoch * length)
                    # warm up learning rate
                    if iterNow<=args.warm_iter+1:
                        optimizer.param_groups[0]['lr'] = args.warmup_lr + (iterNow-1) * (args.lr-args.warmup_lr)/args.warm_iter
                    input, target = data

                    input, target = input.to(device), target.to(device)
                    # clears wi.grad for every weight wi in the optimizer. 
                    optimizer.zero_grad()
                    # forward propagation
                    output = net(input)

                    # warmup_scheduler.dampen() output=35*1000 target=35*1
                    # thetas = head(output,target)
                    # calculate loss by LabelSmoothSoftmaxCE
                    
                    loss = criterion(output, target)
                    '''=center + focal, center need to change to 2 class(haven't test yet)
                    loss_focal = criterion_focal(thetas, target)
                    loss_center = criterion_center(output, target)
                    loss = loss_center + loss_focal
                    '''
                    # loss = criterion(output, target)
                    # backward propagation, will compute the gradient of loss using wi and record at wi.grad
                    loss.backward()
                    # change the learning rate and parameter in specific epoch
                    optimizer.step()       
                    
                    # record the accuracy and loss
                    train_sum_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)                                 
                    total += target.size(0)
                    correct += predicted.eq(target.data).cpu().sum()

                    lr = optimizer.param_groups[0]['lr']       
                     
                    trainLossEpNow = train_sum_loss / (i + 1)
                    # trainAccEpNow = 100. * float(correct) / float(total)
                    trainAccEpNow = 100. * float(correct) / float(total)
                    # print('warmup:%e', warmup_optimizer.param_groups[0]['lr'])
                    print('[epoch:%d, iter:%d] | LR: %e | Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, iterNow, lr, trainLossEpNow,
                             trainAccEpNow))
                    f2.write('%03d  %05d | LR: %e | Loss: %.03f | Acc: %.3f%% '
                             % (epoch + 1, iterNow, lr, trainLossEpNow, trainAccEpNow))
                    f2.write('\n')
                    f2.flush()
                trainAllLoss.append(trainLossEpNow)
                trainAllAcc.append(trainAccEpNow)
                # torch.save(net, 'output/efficientb4_epoch{}.pkl'.format(epoch))
                # torch.save(head, 'output/efficientb4_head_epoch{}.pkl'.format(epoch))

                # 每訓練完一个 epoch 測試一下準確率
                print("Waiting Test!")
                val_sum_loss = 0
                with torch.no_grad():
                    validLossEpNow = 0.0
                    validEpDiff = 0.0
                    correct = 0
                    total = 0
                    for j, data in enumerate(dataloaders_dict['val'], 0):
                        net.eval()
                        input, target = data
                        input, target = input.to(device), target.to(device)
                        output = net(input)
                        # thetas = head(output,target)
                        loss = criterion(output, target)
                        # loss = criterion(output, target)
                        val_sum_loss += loss.item()
                        validLossEpNow = val_sum_loss / (j + 1)
                        optimizer.zero_grad()

                        # 取得分最高的那个類 (output.data的索引)
                        _, predicted = torch.max(output.data, 1)
                        total += target.size(0)
                        correct += (predicted == target).cpu().sum()
                    print('Test classification accuracy%.3f%%' % (100. * float(correct) / float(total)))
                    acc = 100. * float(correct) / float(total)
                    scheduler.step(acc)                                     

                    # 將每次測試結果寫入 acc.txt 文件中
                    if (ii % 1 == 0):
                        print('Saving model......')
                        torch.save(net.state_dict(), '%s/net_%03d.pth' % (args.results_path, epoch + 1))
                        # torch.save(head.state_dict(), '%s/net_head_%03d.pth' % (args.results_path, epoch + 1))
                        # torch.save(net.state_dict(), '%s/net_%03d.pth' % (args.outf, epoch + 1))
                    f.write("EPOCH=%03d,Accuracy= %.3f%%" % (epoch + 1, acc))
                    f.write('\n')
                    f.flush()
                    validAllLoss.append(validLossEpNow)
                    validAllAcc.append(acc)
                    # 記錄最佳測試分類準確率並寫入 best_acc.txt 文件中
                    if acc > best_acc:
                        f3 = open(args.results_path + args.best_txt_path, "w")
                        f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc))
                        f3.close()
                        best_acc = acc
                    
                    # log(trainLossEpNow,
                    #     trainAccEpNow,
                    #     validLossEpNow,
                    #     acc, epoch)

                N = np.arange(0, eps)
                plt.style.use("ggplot")
                plt.figure()
                plt.plot(N, trainAllAcc, label = "train_acc")
                plt.plot(N, trainAllLoss, label = "train_loss") 
                plt.plot(N, validAllAcc, label = "valid_acc")    
                plt.plot(N, validAllLoss, label = "valid_loss")
                plt.title("Accuracy and Loss")
                plt.xlabel("Epoch #")
                plt.ylabel("Loss/Accuracy")
                plt.legend(loc="lower left")
                plt.savefig(args.results_path +args.plot_path)
                amount = args.at
                amount = amount - 0.5
                #測試 threshold 模式
                # TestJet(args, amount, net, dataloaders_dict['easy_val'], epoch) #Hanyu

                #Hanyu
                
                TestJet(args, net, dataloaders_dict['easy_val'], epoch,amount)


            print("Training Finished, TotalEPOCH=%d" % args.epoch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EfficientNet')

    parser.add_argument('--at', type=float, default=0.98, metavar='N', help='') #Hanyu
    parser.add_argument('--input_size', type=int, default=224, metavar='N', help='')
    parser.add_argument('--bs', type=int, default=8, metavar='N', help='')
    parser.add_argument('--lr', type= float, default=1e-3, metavar='N', help='')
    parser.add_argument('--classnum', type=int, default=2, metavar='N', help='')
    parser.add_argument('--max_of_each_class', type=int, default=25000, metavar='N', help='')
    parser.add_argument('--warm_iter', type=int, default=2000, metavar='N', help='') 
    parser.add_argument('--warmup_lr', type=float, default=0.0, metavar='N', help='')
    parser.add_argument('--pre_epoch', type=int, default=0, metavar='N', help='') 
    parser.add_argument('--epoch', type=int, default=20, metavar='N', help='')
    parser.add_argument('--test_model_num', type=int, default=11, metavar='N', help='')
    parser.add_argument('--test_bs', type=int, default=400, metavar='N', help='')
    parser.add_argument('--score_path', type=str, default="./txt/score.txt", metavar='N', help='')
    parser.add_argument('--test_err_img', type=str, default="./txt/misclassified.txt", metavar='N', help='')
    
    #Hanyu #電容+電阻
    parser.add_argument('--data_path', type=str, default="../dataset/JET_C+R_Classification/Training_set", metavar='N', help='') 
    parser.add_argument('--easy_val_data_path', type=str, default="../dataset/JET_C+R_Classification/EasyValidation_set", metavar='N', help='') 
    #Hanyu
    parser.add_argument('--pth_file', type=str, default=".model\test_C+R_haveword_data_baseline", metavar='N', help='')
    #Hanyu

    parser.add_argument('--is_load_check_point', type=bool, default=False, help='')
    parser.add_argument('--test_data_path', type=str, default="./TestData", metavar='N', help='')
    parser.add_argument('--inference_model_path', type=str, default="./inference_model/net_020.pth", help='')
    parser.add_argument('--inference_result_path', type=str, default="./predict_result.txt", metavar='N', help='')


    parser.add_argument('--feature_extract', type=bool, default=False, metavar='N', help='')
    parser.add_argument('--results_path', type=str, default="./model/test_C+R_haveword_data", metavar='N', help='') #Hanyu
    parser.add_argument('--plot_path', type=str, default="/output/plot.png", metavar='N', help='')
    parser.add_argument('--acc_txt_path', type=str, default="/txt/acc.txt", metavar='N', help='')
    parser.add_argument('--log_txt_path', type=str, default="/txt/log.txt", metavar='N', help='')
    parser.add_argument('--best_txt_path', type=str, default="/txt/best_acc.txt", metavar='N', help='')
    parser.add_argument('--wandb_name', type=str, default="jet_artifact_anil", metavar='S', help='wandb name')
    parser.add_argument('--dataset1', type=str, default='Resistance')
    parser.add_argument('--tstdataset', type=str, default='Resistance') 
    parser.add_argument('--tst_txt_name', type=str, default='testScore.txt')

    #Hanyu
    parser.add_argument('--mode', type=str, default='train',choices=['train', 'adapt', 'test'])
    #Hanyu

    args = parser.parse_args()

    # wandb.login()
    # wandb.init(project=args .wandb_name, config = args)

    if args.mode == 'test':
        Load_test(args)
        #TestJet(args, EmotionDetector, dataloaders_dict['test'], 20,0.48)
    elif args.mode == 'train':
        main(args)