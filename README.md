## Install
```bash
conda install -c https://software.repos.intel.com/python/conda/ -c conda-forge onemkl-sycl-blas
conda install -c https://software.repos.intel.com/python/conda/ -c conda-forge mkl-include
pip install -e .
```

## Run
```bash
bash experiments/end2end/A30/apps/run.sh
bash experiments/end2end/A30/cnndailymail/run.sh
```