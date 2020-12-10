# Ads-RecSys-Datasets
This repository is for private use in APEX Lab via NAS, which collects some datasets for Ads &amp; RecSys uses, and provide easy-to-use hdf5 interface.
The datasets are `iPinYou`, `Criteo`, `Avazu`, `Criteo_Challenge`.
The input data is in a `multi-field categorical` format, and the output data is binary.
The hdf5 interfaces are developed by @xueyuan zhao, the easy-access-on-NAS is contributed by @tianyao chen, and @weinan zhang, @try-skycn, @kevinkune make contributions to feature engineering.
This repository will be long-term maintained.

## Download
``Note``: if the tar files end with .partx, you should cat all the .partx files togather and uncompress it. 

https://pan.baidu.com/s/1usnQtW-YodlPUQ1TNrrafw
or
https://drive.google.com/drive/folders/1thXezQbmuS6Q8-AXmrhB0tLM3mybJxVR?usp=sharing

## Basic Usage
If you can access NAS of APEX Lab, you can use the interface and access the data from ``/newNAS/Datasets/MLGroup/Ads-RecSys-Datasets``. See guides about `NAS`.

Then you can import this repository in python directly:

    import sys
    sys.path.append('/your/local/path')
    from datasets import iPinYou, ...
    
    data = iPinYou()
    data.summary()
    train_gen = data.batch_generator('train')
    
We suggest to configure path in `__init__.py`. For example:
    
    # in __init__.py
    import getpass

    config = {}
    
    # some way you can identify which machine it is
    user = getpass.getuser()
    config['user'] = user
    
    if user.lower() == 'your user name in local machine':
        config['env'] = 'cpu'
    else:
        config['env'] = 'gpu'
        
    # when use, e.g., in run.py
    import __init__
    if __init__.config['env'] == 'cpu':
        sys.path.append('/your/local/path')
    else:
        sys.path.append('/NAS/Dataset/Ads-RecSys-Datasets')
    from datasets import iPinYou, Criteo
  
## Datasets
This section will introduce the hdf5 interfaces, and feature engineering in detail.

### hdf5
Most of the cases, the data we processing could be well aligned, i.e. well structured in single table.
Hdf5 is binary format on disk, which has high I/O performance, supports compression, and unlimited serialization size.
@xueyuan zhao has developed an efficient data generator (in python), which can shuffle in-process, adjust postive sample fraction, change validation set size, and so on.

In this solution, a dataset is treated as an object, which can be easily manipulated whether interactive or not. Here `interactive` means interactive programming styles, like `ipython notebook`. Because we don't suggest maintaining all the data in memory.

Besides, a multi-processing module is developed to spped up data processing if you need.

### iPinYou
The feature engineering is contributed by @weinan zhang. 
On his benchmark [make-ipinyou-data](https://github.com/wnzhang/make-ipinyou-data),
we re-organized the feature alignment and removed the `user-tag` feature considering leaky problems [make-ipinyou-data-refined](https://github.com/Atomu2014/make-ipinyou-data).

In general, this dataset contains 16 categorical features:

    ['weekday', 'hour', 'IP', 'region', 'city', 'adexchange', 'domain',
    'slotid', 'slotwidth', 'slotheight', 'slotvisibility', 'slotformat',
    'creative', 'advertiser', 'useragent', 'slotprice']
    
where `slotwidth` and `slotheight` are treated as categorical features because they only have few values,
and `slotprice` is discretized by thesholds 0, 10, 50, and 100. 
Even though the original data log has over 30 features, we don't use all of them:

- some features are unique IDs appearing only once, which does no help in prediction.
- some of them are auction/impression related prices WHICH CANNOT BE USED IN PREDICTION.
- user tags have leaky concerns.

After one-hot encoding, the feature space approximates 900k.

No negative down sampling, and no removing long-tail data. We preserve most of the information in this engineering.

The train/test sets are officially partitioned, `train set size: 15M`, `test set size: 4M`.
`Positive sample ratio` is `0.00075` on train set, and `0.00073` on test set.

### Criteo
The feature engineering is contributed by @tianyao chen, and he has done most of the works including hdf access in [APEXDatasets](https://github.com/try-skycn/APEXDatasets). 
His work is collected in this repository with some issues fixed and better wrapped.

The original dataset is know as `Criteo 1TB click log`, in which the CriteoLab has collected 30 days of masked data.
We only know there are 13 numerical and 26 categorical features, and there is no feature description released.
Thus we name thease features as `num_0 ... num_12`, and `cat_0 ..., cat_25`.

For numerical features, @tianyao chen discretized them by equal-size buckets, referring [APEXDatasets](https://github.com/try-skycn/APEXDatasets). 

For categorical features, he removed long-tailed data appearing less than 20 times.

Nagetive sown sampling is used, and the resulting positive sample ratio is about 0.5.

After one-hot encoding, the feature space approximates 1M.

The train/test sets are partitioned by us. We use one week data to train, and following 1 day to test.
`train set size: 86M`, `test set size: 12M`.
`Positive sample ratio` is `0.50` on train set, and `0.49` on test set.

### Avazu

The raw dataset can be downloaded from https://www.kaggle.com/c/avazu-ctr-prediction/data.

### Criteo Challenge

We follow the data processing of https://github.com/guestwalk/kaggle-2014-criteo (master branch), and convert the features into hdf format directly.

**update** original log download link: https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/
