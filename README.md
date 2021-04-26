# hadWChargeTagger

## Installation
* Download miniconda for python3 from https://docs.conda.io/en/latest/miniconda.html
* run ```bash Miniconda3-latest-Linux-x86_64.sh``` to start the installation.
* Makesure you don't have preset path variables like PYTHONPATH etc. Also, do this outside of CMSSW.
* your ~/.bashrc will be edited and path variables will be automatically set up.
* For changes to take effect, close and re-open your current shell.

If you'd prefer that conda's base environment not be activated on startup, 
set the auto_activate_base parameter to false: 

```conda config --set auto_activate_base false```

* Setup environment for training using TensorFlow/Keras ```conda env create -f env/environment.yml```
* To activate environment:```conda activate tf```
* To deactivate environment: ```conda deactivate```
* To see the list of your envs: ```conda info --envs```
* To change your current environment back to the default (base): ```conda activate```
