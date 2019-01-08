# Neural Local Coherene Model for Text Quality Assessment #

A neural local coherence model based on semantic changes across sentences in a text. 

### Set Up ###

* Install PyTorch
* Prepare data
* Run main_making.py

### Data ###

For the essay scoring data, we use the ASAP dataset to evaluate our system. This dataset (training_set_rel3.tsv) can be downloaded from [here](https://www.kaggle.com/c/asap-aes/data). After downloading the file, put it in the [data](https://github.com/nusnlp/nea/tree/master/data) directory and create training, development and test data using ```preprocess_asap.py``` script:

```bash
cd data
python preprocess_asap.py -i training_set_rel3.tsv
```

### Publication ###

Mohsen Mesgar and Michael Strube. 2018. [A Neural Local Coherence Model for Text Quality Assessment](http://aclweb.org/anthology/D18-1464). In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing. 
