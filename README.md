# supervised-grns
Supervised inference of gene regulatory networks from single-cell RNA-sequencing data

## Installation

### Download the code on your machine
```git clone https://github.com/Murali-group/supervised-grns.git```

### Setting up conda environment and dependencies
We recommend using an [Anaconda](https://www.anaconda.com/) environment to run the pipeline. The below steps can be followed on your Ubuntu machine to get the pipeline setup and running.
#### Install conda and activate the environment
* Install Anaconda from the official website [here](https://www.anaconda.com/products/individual#Downloads)
* After you have Anaconda installed on your machine, we will create a conda environment specific for this project. This project works on ```Python3```
  ```bash
  conda create -n "SGRN" python=3.7.10 ipython
  ```
  * This will create an environment called ```SGRN``` with ```Python 3.7.10```
  * Activate the environment with the command 
    ```bash
    conda activate SGRN
    ```
  * We will now install the required packages and dependencies in the ```SGRN``` environment.
#### Install dependencies
* supervised-grns pipeline mainly uses ```PyTorch``` framework for its computation, amongst other libraries. We found that installing ```PyTorch``` can be tricky so we split this step into two - 
  * Install ```PyTorch``` and related modules
  * Install rest of the packages through ``` pip install requirements.txt```


* In this step, we will install ```PyTorch``` library along with the necessary dependencies.
  * Install PyTorch in your environment with the following command - ```conda install -c pytorch pytorch```
  * Check if PyTorch is installed and check the version. You should get an output like this -
    ``` bash 
    python -c "import torch; print(torch.__version__)"
    1.8.0
    ```
  * Similarly, install ```torchvision``` and check if is correctly installed -
    ``` bash
    conda install -c pytorch torchvision
    python -c "import torchvision; print(torchvision.__version__)"
    0.9.0
    ```
  * Finally, the ```PyTorch geometric``` library can be installed by following the steps [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) listed under __Installation via Binaries__
  
* For the evaluation, we additionally need R packages. For this install R within the SGRN conda environment using:
  *   ```conda install r=3.5.0```
  *   ```R -e "install.packages('https://cran.r-project.org/src/contrib/PRROC_1.3.1.tar.gz', type = 'source')"```
* We now install the rest of the libraries using the ```requirements.txt``` file.
  ```bash 
  pip install requirements.txt
  ```
That's it! We can now run the pipeline and check if everything is working fine.

## Run the pipeline

Navigate to your folder where the code is downloaded. Run the following command to test the running of the pipeline.
```bash
 python main.py --config=config/config.yaml 
 ```

