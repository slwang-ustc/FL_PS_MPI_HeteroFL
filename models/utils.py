import sys
sys.path.append('/data/slwang/FL_PS_MPI_HeteroFL/models')
from cnn import *


def create_model_instance(model_type, model_ratio, track=False):
    if model_type == 'cnn_hetero':
        return create_cnn(model_ratio=model_ratio, track=track)
    else:
        raise ValueError("Not valid model type")
