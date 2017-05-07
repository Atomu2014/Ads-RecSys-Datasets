from .Criteo import Criteo
from .iPinYou import iPinYou


def as_dataset(data_name, **kwargs):
    data_name = data_name.lower()
    if data_name == 'criteo':
        return Criteo(**kwargs)
    elif data_name == 'ipinyou':
        return iPinYou(**kwargs)
