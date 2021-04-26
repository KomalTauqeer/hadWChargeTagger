# hadWChargeTagger

## Installation
* Download miniconda for python3 from https://docs.conda.io/en/latest/miniconda.html
* bash Miniconda3-latest-Linux-x86_64.sh to start installation
* Makesure you don't have preset path variables like PYTHONPATH etc. Also, do this outside of CMSSW.
* your ~/.bashrc will be edited like this:

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/ktauqeer/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/ktauqeer/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/ktauqeer/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/ktauqeer/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

# ==> For changes to take effect, close and re-open your current shell. <==

If you'd prefer that conda's base environment not be activated on startup, 
   set the auto_activate_base parameter to false: 

conda config --set auto_activate_base false

* Setup environment for training using TensorFlow/Keras ```conda env create -f env/environment.yml```
* Activate environment ```conda activate tf```
* Activate environment ```conda deactivate```
* To see the list of your envs:
conda info --envs
* To change your current environment back to the default (base): ``conda activate````
