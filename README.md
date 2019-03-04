# A Neural Local Coherene Model for Text Quality Assessment #

A neural local coherence model based on semantic changes across sentences in a text. Your interest to this project is very appreciated. Pease cite [this paper](https://aclanthology.info/papers/D18-1464/d18-1464.bib) if you use the above code. 
Also don't forget to give it a star, if you like the project.


### Setup ###
* OS: linux2
* GCC 7.2.0
* Python 2.7.14 
* PyTorch 0.1.12_2
* More info: [spec-file.txt](https://github.com/MMesgar/neural_coherence_model/blob/master/spec-file.txt)

### Procedure ###
* Prepare data
* Run main_making.py

### Data ###

For the essay scoring experiments, we use the ASAP dataset to evaluate our system. This dataset (training_set_rel3.tsv) can be downloaded from [here](https://www.kaggle.com/c/asap-aes/data). After downloading the file, put it in the [data](https://github.com/MMesgar/neural_coherence_model/tree/master/data) directory and create training, development and test data using ```preprocess_asap.py``` script:

```bash
cd data
python preprocess_asap.py -i training_set_rel3.tsv
```

### Publication ###

Mohsen Mesgar and Michael Strube. 2018. [A Neural Local Coherence Model for Text Quality Assessment](http://aclweb.org/anthology/D18-1464). In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing. 
