from Autoformer import Autoformer
from DLinear import DLinear
from ETSformer import ETSformer
from Informer import Informer
from Linear import Linear
from NLinear import NLinear
from Pyraformer import Pyraformer
from RAFT import RAFT
from Stat_models import Naive_repeat,Arima,SArima,GBRT
from TemporalFlow import TemporalFlow
from Transformer import Transformer
from WAVESTATE import WAVESTATE


def get_model(configs):
    if configs.model == 'Autoformer':
        return Autoformer(configs)
    elif configs.model == 'DLinear':
        return DLinear(configs)
    elif configs.model == 'ETSformer':
        return ETSformer(configs)
    elif configs.model == 'Informer':
        return Informer(configs)
    elif configs.model == 'Linear':
        return Linear(configs)
    elif configs.model == 'NLinear':
        return NLinear(configs)
    elif configs.model == 'Pyraformer':
        return Pyraformer(configs)
    elif configs.model == 'RAFT':
        return RAFT(configs)
    elif configs.model == 'TemporalFlow':
        return TemporalFlow(configs)
    elif configs.model == 'Transformer':
        return Transformer(configs)
    elif configs.model == 'WAVESTATE':
        return WAVESTATE(configs)
    else:
        raise ValueError('Model not found')

__all__ = ['get_model', 
'Autoformer', 'DLinear', 
'ETSformer', 'Informer', 
'Linear', 'NLinear', 
'Naive_repeat', 'Arima', 'SArima', 'GBRT',
'Pyraformer', 'RAFT', 
'TemporalFlow', 'Transformer', 
'WAVESTATE']