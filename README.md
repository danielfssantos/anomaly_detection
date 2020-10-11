# Enhancing Anomaly Detection Through Restricted Boltzmann Machine Features Projection

*This repository holds all the necessary code to run the very-same experiments described in the paper "Enhancing Anomaly Detection Through Restricted Boltzmann Machine Features Projection".*

---

## References

If you use our work to fulfill any of your needs, please cite us:

```
```

---

## Package Guidelines

### Installation

1. Download libsvm from [here](http://www.csie.ntu.edu.tw/~cjlin/cgi-bin/libsvm.cgi?+http://www.csie.ntu.edu.tw/~cjlin/libsvm). Follow the libsvm README build steps and also remember to build the python interface.

2. Substitute the python libsvm path inside `svm_detector.py`:
```bash
sys.path.append("path_to_libsvm_folder/python")
```

3. Create an environment, activate it, and install dependencies:
```bash
python3.6 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

### Executing the code

After installing the necessary components, execute all the training and test procedures:

```bash
sudo chmod 777 ./*.sh
./run_experiments.sh
```

### Generating Results

After executing `run_experiments.sh`, a `results/` folder will be generated, where one can find all the quantitative and qualitative generated results.

Obs: Be aware that RBMs stochastic nature may produce slightly different results from the paper's ones.

---

## Support

We know that we do our best, but it is inevitable to acknowledge that we make mistakes. If you ever need to report a bug, report a problem, talk to us, please do so! We will be available at our bests at this repository or danielfssantos1@gmail.com, gustavo.rosa@unesp.br, and mateus.roder@unesp.br.

---
