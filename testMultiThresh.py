from sklearn import metrics
import torch
from torchvision import datasets, transforms
import os
import cv2
import config as cfg
import logging
import matplotlib.pyplot as plt
import numpy as np
import itertools
from imutils import paths
from sklearn.metrics import roc_curve, auc, confusion_matrix
import torchvision.transforms.functional as TF
from dataset.customDataJet import customData
from util.utils import my_collate_fn
from torch.utils.data.dataloader import default_collate  
import csv
import wandb
import sys


from torch.nn import DataParallel
from misc.utils import get_inf_iterator, mkdir
plt.rcParams["axes.grid"] = False


sys.path.append('model')
sys.path.append('../')

print(sys.path)
# def read_image(test_data_path):
#     img = cv2.imread(test_data_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     return img

#Hanyu
Gobal_T = 0
#Hanyu

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # plt.tight_layout()    

'''
def model_eval(actual, pred, modelIdx):
    actual = list(map(lambda el:[el], actual))
    pred = list(map(lambda el:[el], pred))
    cm = confusion_matrix(actual, pred)
    plt.figure()
    labelName = ['OK', 'NG']
    plot_confusion_matrix(cm, classes=labelName, normalize=True,
                    title="confusion matrix")
    # plt.savefig(""cfg.CONFUSION_PATH)
    plt.savefig("%s/CM_norm%03d.png" %(cfg.MODEL_PATH, modelIdx))
    
    plt.figure()
    plot_confusion_matrix(cm, classes=labelName, normalize=False,
                    title="confusion matrix")
    plt.savefig("%s/CM_%03d.png" %(cfg.MODEL_PATH, modelIdx))
    
    # plt.show()
    # print(cm)
'''
def model_eval(args, actual, pred, predsDecimal, modelIdx, amount):
    with open(args.results_path + "/txt/"+args.tstdataset+"_"+args.tst_txt_name,"a+") as f:
        
        # calculate eer

        fpr, tpr, threshold = roc_curve(actual,predsDecimal)          
        fnr = 1-tpr
        diff = np.absolute(fnr - fpr) #陣列取絕對值
        idx = np.nanargmin(diff)      #在有Nan值，選擇最小的array的index
        # print(threshold[idx])
        eer = np.mean((fpr[idx],fnr[idx]))        

        avg = np.add(fpr, fnr)
        idx = np.nanargmin(avg)
        hter = np.mean((fpr[idx],fnr[idx])) #取出陣列fpr、fnr最小值，算出平均

        fpr_at_10e_m3_idx = np.argmin(np.abs(fpr-10e-3))
        tpr_cor_10e_m3 = tpr[fpr_at_10e_m3_idx+1]

        fpr_at_5e_m3_idx = np.argmin(np.abs(fpr-5e-3))
        # print(fpr[-1])
        tpr_cor_5e_m3 = tpr[fpr_at_5e_m3_idx+1]

        fpr_at_10e_m4_idx = np.argmin(np.abs(fpr-10e-4))
        tpr_cor_10e_m4 = tpr[fpr_at_10e_m4_idx+1]

        # print(actual)   #actual 是lable完的list
        # print(pred)     #pred   是predict完的list
        actual = list(map(lambda el:[el], actual))
        pred = list(map(lambda el:[el], pred))
        
        cm = confusion_matrix(actual, pred)
        TP = cm[0][0]
        TN = cm[1][1]
        FP = cm[1][0]
        FN = cm[0][1]

    
        labelName = ['OK', 'NG']
        
        plt.figure()
        plt.rcParams["axes.grid"] = False
        plot_confusion_matrix(cm, classes=labelName, normalize=True,
                        title="confusion matrix")
        # plt.savefig(""cfg.CONFUSION_PATH)


        confusion_save_path = os.path.join(args.results_path, "confusion")
        mkdir(confusion_save_path)
        plt.savefig("%s/CM_norm%03d.png" %(confusion_save_path, modelIdx))
        
        plt.figure()
        plt.rcParams["axes.grid"] = False
        plot_confusion_matrix(cm, classes=labelName, normalize=False,
                        title="confusion matrix")
        plt.savefig("%s/CM_%03d.png"  %(confusion_save_path, modelIdx))

        amount =amount +0.5
        accuracy = ((TP+TN))/(TP+FN+FP+TN)
        precision = (TP)/(TP+FP)
        recall = (TP)/(TP+FN)
        f_measure = (2*recall*precision)/(recall+precision)
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)		
        error_rate = 1 - accuracy
        apcer = FP/(TN+FP)
        bpcer = FN/(FN+TP)
        acer = (apcer+bpcer)/2
        f.write("="*60)
        f.write('\nModel %03d \n'%(modelIdx))
        f.write('TP:%d, TN:%d,  FP:%d,  FN:%d\n' %(TP,TN,FP,FN))
        f.write('threshold:%f\n'%(amount))
        f.write('accuracy:%f\n'%(accuracy))
        f.write('precision:%f\n'%(precision))
        f.write('recall:%f\n'%(recall))
        f.write('f_measure:%f\n'%(f_measure))
        f.write('sensitivity:%f\n'%(sensitivity))
        f.write('specificity:%f\n'%(specificity))
        f.write('error_rate:%f\n'%(error_rate))
        f.write('apcer:%f\n'%(apcer))
        f.write('bpcer:%f\n'%(bpcer))
        f.write('acer:%f\n'%(acer))
        f.write('eer:%f\n'%(eer))
        f.write('hter:%f\n'%(hter))
        f.write('TPR@FPR=10E-3:%f\n'%(tpr_cor_10e_m3))
        f.write('TPR@FPR=5E-3:%f\n'%(tpr_cor_5e_m3))
        f.write('TPR@FPR=10E-4:%f\n\n'%(tpr_cor_10e_m4))
        wandb.log({"apcer": apcer, "bpcer": bpcer, "acer": acer, "eer": eer, "hter": hter,
                    "accuracy_test": accuracy, "error_rate":error_rate, "precision": precision, "recall": recall, "f_measure": f_measure, 
                    "sensitivity":sensitivity, "specificity":specificity,
                    "TPR@FPR=10E-3": tpr_cor_10e_m3, "TPR@FPR=5E-3": tpr_cor_5e_m3, "TPR@FPR=10E-4": tpr_cor_10e_m4},step=modelIdx)
     


def TestJet(args, amount, model, data_loader_target, modelIdx):

    # print("***The type of norm is: {}".format(normtype))
    if not os.path.isdir(args.results_path + "/txt"):
        os.mkdir(args.results_path + "/txt")

    ####################
    # 1. setup network #
    ####################
    # set train state for Dropout and BN layers    
    model.eval()

    if torch.cuda.device_count() > 1:
        model = DataParallel(model)
        
    #Hanyu
    # score_list = []
    # label_list = []
    # pred_list = []
    #Hanyu
    
    limitRecordNums = 50
    mis_cls_ok_as_ng_img = []
    mis_cls_ok_as_ng_path = []    

    mis_cls_ng_as_ok_img = []
    mis_cls_ng_as_ok_path = []
    #Hanyu
    
    # idx = 0
    #Hanyu
    # def calcDiffThreshPred(output,target):
    #     m = torch.nn.Softmax(dim=1)
    # 
    # thresh = -0.3
    # while thresh<=0.3:
    #     temp_probs = probs+thresh
    #     preds = torch.round(temp_probs)
    #     print(preds)
    #     print("*"*50)
    #     thresh+=0.05

    # def calcDiffThreshPred(amount):
    #init
    idx = 0
    while amount<=0.5:
        score_list = []
        label_list = []
        pred_list = []
        with torch.no_grad():
            with open(args.results_path + "/txt/misclassified.txt","a+") as mis_f:
                mis_f.write("="*60)
                mis_f.write('\nModel %03d'%(modelIdx))
                #Hanyu
                for (img_paths, catimages, labels) in data_loader_target:
                    images = catimages.cuda()
                    label_pred  = model(images)
                    
                    # score = torch.sigmoid(label_pred).cpu().detach().numpy()
                    m = torch.nn.Softmax(dim=1)
                    score = m(label_pred)
                
                    score = score[:,1:]
                
                    # print(probs)
                    # score = np.squeeze(score[:,1:], 1)
                    # print(score)
                    #Hanyu
                    score = score + amount

                    #Hanyu
                    pred = np.round(score.cpu().numpy())
                    pred = np.array(pred, dtype=int)
                    pred = np.squeeze(pred, 1)

                    pred_list = pred_list + pred.tolist()
                    labels = labels.tolist()

                    score_list = score_list + score.tolist()
                    label_list = label_list + labels
                    

                    # print('SampleNum:{} in total:{}, score:{}'.format(idx,len(data_loader_target), score.squeeze()))
                    idx+=1
                    
                    # 記錄分類錯誤的
                    for i in range(len(score)):
                        pd = pred[i]
                        gt = labels[i]
                        if gt!=pd:
                            img = images[i]
                            imgPath = img_paths[i]
                            mis_f.write('GT:%06d,  PD:%06d,  Path:%s\n'%(gt, pd, imgPath))                    
                            mis_f.flush()
                            if gt == 0 and pd == 1 and len(mis_cls_ok_as_ng_img)<limitRecordNums:
                                mis_cls_ok_as_ng_img.append(img)
                                mis_cls_ok_as_ng_path.append(imgPath)
                                # wandb.log({"GT:ok，PD:ng": wandb.Image(img, caption=imgPath)})
                            elif gt == 1 and pd ==0 and len(mis_cls_ng_as_ok_img)<limitRecordNums:
                                mis_cls_ng_as_ok_img.append(img)
                                mis_cls_ng_as_ok_path.append(imgPath)

            # wandb.log({"Misclassfied ok as ng": [wandb.Image(image, caption=path) for image, path in zip(mis_cls_ok_as_ng_img, mis_cls_ok_as_ng_path)]}, step=modelIdx)
            # wandb.log({"Misclassfied ng as ok": [wandb.Image(image, caption=path) for image, path in zip(mis_cls_ng_as_ok_img, mis_cls_ng_as_ok_path)]}, step=modelIdx)
            
            fpr, tpr, _ = roc_curve(label_list, score_list)

            fnr = 1-tpr
            if args.dataset1 == args.tstdataset: # inter dataset
                roc_auc = auc(fpr, tpr) #x,y
                plot_score(args, modelIdx, fpr, tpr, fnr, roc_auc, cross_data = False, log = False)
                plot_score(args, modelIdx, fpr, tpr, fnr, roc_auc,  cross_data = False, log = True)
                roc_auc = auc(fpr, fnr) #x,y
                plot_score(args, modelIdx, fpr, tpr, fnr, roc_auc, cross_data = True, log = False)
                plot_score(args, modelIdx, fpr, tpr, fnr, roc_auc, cross_data = False, log = False)
            
            # plot_score()
            # model_eval(args, label_list, pred_list, score_list, modelIdx)
    
        model_eval(args, label_list, pred_list, score_list, modelIdx, amount)
        print(amount)
        if (amount+0.5) < 0.84:
            amount +=0.05
        elif (amount+0.5) >= 0.85 and (amount+0.5)<0.95:
            amount +=0.03
        elif (amount+0.5)>= 0.95 and (amount+0.5)<0.99:
            amount +=0.01
        elif (amount+0.5)> 0.99:
            amount +=0.005
    
        print('threshold = :',amount+0.5)
        



def plot_score(args, modelIdx,fpr, tpr, fnr, roc_auc, cross_data = False, log = False):
    fig = plt.figure()
    lw = 2

    if not cross_data:
        if log:
            plt.xscale("log")
        elif not log:
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)
        plt.ylabel('True Living Rate')
    elif cross_data:
        if log:
            plt.xscale("log")
        elif not log:
            plt.plot([0, 1], [1, 0], color='navy', lw=lw, linestyle='--')#(x0,x1), (y0,y1)
        plt.plot(fpr, fnr, color='darkorange', lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)
        plt.ylabel('False Fake Rate')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Living Rate')

    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    #fig.savefig('/tmp/roc.png')
    curve_save_path = os.path.join(args.results_path, "curve")
    mkdir(curve_save_path)
    if log:
        if cross_data:
            plt.savefig("%s/ROC_cross_log_%s_%03d.png" %(curve_save_path, args.tstdataset, modelIdx))
        elif not cross_data:
            plt.savefig("%s/ROC_log_%s_%03d.png" %(curve_save_path, args.tstdataset, modelIdx))
    elif not log:
        if cross_data:
            plt.savefig("%s/ROC_cross_%s_%03d.png" %(curve_save_path, args.tstdataset, modelIdx))
        elif not cross_data:
            plt.savefig("%s/ROC_%s_%03d.png" %(curve_save_path, args.tstdataset, modelIdx))