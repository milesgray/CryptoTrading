from .forecast import ForecastExp
from .movement import MovementExp

def get_exp(configs):
    if configs.task_name == 'forecast':
        return ForecastExp(configs)
    elif configs.task_name == 'movement':
        return MovementExp(configs)
    else:
        raise ValueError('Task name not found')

__all__ = [
    "get_exp",
    "ForecastExp",
    "MovementExp",
]