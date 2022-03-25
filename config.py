import os
INPUT_SIZE = 224
# BS = 60
BS = 10
LR = 1e-3 
CLASS_NUM = 2
# MAX_OF_EACH_CLASS = 4250
# MAX_OF_EACH_CLASS = 10000
# MAX_OF_EACH_CLASS = 15000
MAX_OF_EACH_CLASS = 25000
WARM_ITER = 2000
WARMUP_LR = 0.0
PRE_EPOCH = 0
EPOCH = 5
TEST_MODEL_NUM = 11
TEST_BS = 400
SCORE_PATH = "./txt/score.txt"
TEST_ERR_IMG = "./txt/misclassified.txt"

DATA_PATH = "..\Jet_File_Classification_version1\Training_set" #電阻
EASY_VAL_DATA_PATH = "..\Jet_File_Classification_version1\EasyValidation_set" #電阻

# DATA_PATH = "../../JET_C_Classification/Training_set" #電容
# EASY_VAL_DATA_PATH = "../../JET_C_Classification/EasyValidation_set" #電容
FEATURE_EXTRACT = False


MODEL_PATH = "./model/R_2_classes_wo_word" #電阻
#MODEL_PATH = "./model/C_2_classes_wo_word" #電容


# MODEL_PATH = "./temp"
PLOT_PATH = "/output/plot.png"
ACC_TXT_PATH = "/txt/acc.txt"
LOG_TXT_PATH = "/txt/log.txt"
BEST_TXT_PATH = "/txt/best_acc.txt"