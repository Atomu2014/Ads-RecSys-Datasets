# Ads-RecSys-Datasets
This repository collects some datasets for Ads &amp; RecSys uses, and provide easy-to-use hdf5 iterative access.
The datasets are `iPinYou`, `Criteo`, `MovieLens`, `NetFlix`, and `Yahoo Music`.
This repository is to introduce a more standard feature engineering, mainly focus on `multi-field categorical data`, or even `multi-value data` (or `set data`).
The hdf5 interfaces are developed by @xueyuan zhao, the easy-access-on-NAS is contributed by @tianyao chen, and @weinan zhang, @try-skycn, @minzheniu, @kevinkune make contributions to feature engineering.
This repository will be long-term maintained.

## Basic Usage
This repository is located at /NAS/Dataset.
  
    sudo mount -t nfs 172.16.2.30:/mnt/NAS/Dataset/Ads-RecSys-Datasets /your/local/path

Then you can import this repository in python directly:

    import sys
    sys.path.append('/your/local/path')
    from datasets import iPinYou, ...
  
We may support python3 access in the future.

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

### MovieLens

already done, need to update

### NetFlix

already done, need to update

### Yahoo Music

already done, need to update
