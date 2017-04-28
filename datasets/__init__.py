from .Criteo import Criteo
from .iPinYou import iPinYou


def as_dataset(data_name, **kwargs):
    if data_name.lower() == 'criteo':
        return Criteo(**kwargs)
    elif data_name.lower() == 'ipinyou':
        return iPinYou(**kwargs)
