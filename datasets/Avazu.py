from Dataset import Dataset


class Avazu(Dataset):
    block_size = None
    train_num_of_parts = 0
    test_num_of_parts = 0
    train_size = 0
    test_size = 0
    train_pos_samples = 0
    test_pos_samples = 0
    train_neg_samples = 0
    test_neg_samples = 0
    train_pos_ratio = 0
    test_pos_ratio = 0
    initialized = 0
    max_length = 0
    num_features = 0
    feat_names = None
    feat_min = None
    feat_sizes = None
    raw_data_dir = None
    feature_data_dir = None
    hdf_data_dir = None

    def raw_to_feature(self, **kwargs):
        pass
