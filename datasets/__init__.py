from .Criteo import Criteo
from .iPinYou import iPinYou
from .Avazu import Avazu


def as_dataset(data_name, **kwargs):
    data_name = data_name.lower()
    if data_name == 'criteo':
        return Criteo(**kwargs)
    elif data_name == 'ipinyou':
        return iPinYou(**kwargs)
    elif data_name == 'avazu':
        return Avazu(**kwargs)
