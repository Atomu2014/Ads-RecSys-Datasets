from .Criteo import Criteo
from .iPinYou import iPinYou
from .Avazu import Avazu
# from .Huawei import Huawei
from .Criteo_all import Criteo_all
from .Criteo_Challenge import Criteo_Challenge


def as_dataset(data_name, initialized=True):
    data_name = data_name.lower()
    if data_name == 'criteo':
        return Criteo(initialized=initialized)
    elif data_name == 'ipinyou':
        return iPinYou(initialized=initialized)
    elif data_name == 'avazu':
        return Avazu(initialized=initialized)
    elif data_name == 'criteo_9d':
        return Criteo_all(initialized=initialized, num_of_days=9)
    elif data_name == 'criteo_16d':
        return Criteo_all(initialized=initialized, num_of_days=16)
    elif data_name == 'criteo_challenge':
        return Criteo_Challenge(initialized=initialized)
    # elif data_name == 'huawei':
    #     return Huawei(initialized=initialized)
