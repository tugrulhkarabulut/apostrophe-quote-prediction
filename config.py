from yacs.config import CfgNode as CN


_C = CN()
_C.INPUT = "./data/"
_C.OUTPUT = "./output/"
_C.MODEL = "LSTM"
_C.TEST_SPLIT = 0.1

_C.DATA = CN()
_C.DATA.DATASET_NAME = 'tokenized_dataset_bert'
_C.DATA.LOAD_DATASET_FROM_DISK = False


_C.LSTM = CN()
_C.LSTM.HIDDEN_DIM = 256
_C.LSTM.MAX_LEN = 500
_C.LSTM.SOLVER = CN()
_C.LSTM.SOLVER.LR = 1e-3
_C.LSTM.SOLVER.PATIENCE = 5
_C.LSTM.SOLVER.BATCH_SIZE = 128
_C.LSTM.SOLVER.EPOCHS = 10


_C.TRANSFORMER_SOLVER = CN()
_C.TRANSFORMER_SOLVER.LR = 3e-5
_C.TRANSFORMER_SOLVER.TRAIN_BATCH_SIZE = 128
_C.TRANSFORMER_SOLVER.TEST_BATCH_SIZE = 128
_C.TRANSFORMER_SOLVER.GRAD_ACC_STEPS = 2
_C.TRANSFORMER_SOLVER.GRAD_CKPT = True
_C.TRANSFORMER_SOLVER.FP16 = True
_C.TRANSFORMER_SOLVER.EPOCHS = 10
_C.TRANSFORMER_SOLVER.WEIGHT_DECAY = 1e-2

_C.BERT = CN()
_C.BERT.BACKBONE = "bert-base-cased"
_C.BERT.LOAD_DATASET_FROM_DISK = False
_C.BERT.RESUME_FROM_CKPT = False
_C.BERT.CKPT_PATH = ''

_C.T5 = CN()
_C.T5.BACKBONE = "t5-small"
_C.T5.LOAD_DATASET_FROM_DISK = False
_C.T5.RESUME_FROM_CKPT = False
_C.T5.CKPT_PATH = ''



def get_cfg_defaults():
    return _C.clone()
