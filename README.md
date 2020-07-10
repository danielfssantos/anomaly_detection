# Enhancing Anomaly Detection Through Restricted Boltzmann Machine Features Projection

Code implementation of the work Enhancing Anomaly Detection Through Restricted Boltzmann Machine Features Projection. To properly execute the code follow the steps below:

## Install components

1. Download libsvm from [here](ttp://www.csie.ntu.edu.tw/~cjlin/cgi-bin/libsvm.cgi?%20http://www.csie.ntu.edu.tw/~cjlin/libsvm%20tar.gz). Follow the libsvm README build steps and also remember to build the python interface.

2. Substitute the python libsvm path inside svm_detector.py by:
```bash
sys.path.append("path_to_libsvm_folder/python")
```

3. Create an environment, activate it and install python dependencies:
```bash
python3.6 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Execute the code

After installing the necessary components, to execute all the training and test procedures do:

```bash
sudo chmod 777 ./*.sh
./run_experiments.sh
```

## Results

After executing run_experiments.sh script a folder named Results will be generated. Inside such folder one can find all the quantitative and qualitative generated results.

Obs: Be aweare that the stochastic nature of the RBM training may produce results slightly different from the ones in the journal.
