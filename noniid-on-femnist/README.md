## Datasets

### FEMNIST

  * **Overview:** Image Dataset
  * **Details:** 62 different classes (10 digits, 26 lowercase, 26 uppercase), images are 28 by 28 pixels (with option to make them all 128 by 128 pixels), 3500 users
  * **Task:** Image Classification

## Notes

- Install the libraries listed in ```requirements.txt```
    - I.e. with pip: run ```pip3 install -r requirements.txt```
- Go to directory of respective dataset for instructions on generating data
    - in MacOS check if ```wget``` is installed and working
- `*_model` directories contain instructions on running  our proposed method and other benchmark algorithm implementations
  * `geomed_model` is the implementation of `Geometric median`
  * `med_model` is the implementation of `Median`
  * `rsa_model` is the implementation of `RSA`
  * `sgd_model` is the implementation of `SGD`
  * `stochastic_admm_model` is the implementation of `Stochastic ADMM` (Our method)
 
