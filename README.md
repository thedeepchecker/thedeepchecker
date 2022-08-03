# TheDeepChecher

TheDeepChecher is a Python Framework for debugging DL programs (written with Tensorflow).
It is a proof of concept implementation of a reseach paper on the property-based testing for DL programs.
(To add #Link to the paper)

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install TheDeepChecher.

```bash
git clone https://github.com/hoss-bb/thedeepchecker
pip install -r ./thedeepchecker/requirements.txt 
pip install ./thedeepchecker
```

## Usage

You can find a lot of examples on the different folders: *clean_base_programs*, *buggy_synthetic_programs*, *SO_buggy_examples*, *GH_buggy_examples*.
As an example, I will comment out, in the following, the snippets of code added to run TheDeepChecker taking from *clean_base_programs/baseCNN.py*

```python
from deep_checker.checkers import DeepChecker
import deep_checker.interfaces as interfaces
import deep_checker.data as data

#Create here data loaders using the arrays of problem data as they will be loaded to the training algorithm with some common settings of ML/DL data loaders.
data_loader_under_test = data.DataLoaderFromArrays(x_train, y_train, shuffle=True, one_hot=True, normalization=True)
test_data_loader = data.DataLoaderFromArrays(x_test, y_test, shuffle=True, one_hot=True, normalization=True)

#Your own  model class 
model = Model()
#Create a model interface that would be used by TheDeepChecker as a connection to different components on your original model with some specifications required for the validation of some properties
model_under_test = interfaces.build_model_interface(problem_type='classification', 
                                                    test_mode=False,
                                                    features=model.features, 
                                                    targets=model.labels, 
                                                    train_op=model.train_op, 
                                                    loss=model.loss, 
                                                    reg_loss=model.reg_loss,
                                                    perf=model.acc, 
                                                    perf_dir='Up', 
                                                    outs_pre_act=model.logits, 
                                                    outs_post_act=model.probabilities)
#Create a data interface to the data loaders, in fact, the data loaders could be yours with only a need for some required functions that you can find in the parent class depending whether there is an augmentation or not: deep_checker.data.DataLoader or deep_checker.data.AugmentedDataLoader
data_under_test = interfaces.build_data_interface(data_loader_under_test, test_data_loader, homogeneous=True)
#Create an instance of DeepChecker connected to the data and model interfaces previously-created (FYI,there are some parameters to set up if needed).
checker = DeepChecker(name='clean_baseCNN', data=data_under_test, model=model_under_test, buffer_scale=10)
#Here, we are calling all the debugging phases' checks supported including: PreChecks, OverfitChecks, and PostChecks (FYI, there are some parameters to set up if needed as well as other methods to call explicitly one of the debugging phase's checks).
checker.run_full_checks()
```

## Setup your Custom Configuration

First way is to copy-paste the configuration by default, then, replace the settings' options as you want.

```bash
cd path_to/your_DL_programs_dir
mkdir config
cp TheDeepChecker/deep_checker/config/settings.yaml path_to/your_DL_programs_dir/config/settings.yaml
vim settings.yaml
```
Second method is to create an empty file yaml, then override only the settings' options that you want to modify with respect to the settings file's tree structure.
In the following example of partial settings.yaml, I will disable the initial weigths precheck while keeping all the other settings by default.

```yaml
PreCheck:
    fail_on: false
    disabled: false
    Initial_Weight:
        disabled: true
```

## The Paper
You can find the paper [here]([https://dl.acm.org/doi/abs/10.1145/3470006](https://dl.acm.org/doi/abs/10.1145/3529318)) and the citation is as follows:

    @article{10.1145/3529318,
      author = {Braiek, Houssem Ben and Khomh, Foutse},
      title = {Testing Feedforward Neural Networks Training Programs},
      year = {2022},
      publisher = {Association for Computing Machinery},
      address = {New York, NY, USA},
      issn = {1049-331X},
      url = {https://doi.org/10.1145/3529318},
      doi = {10.1145/3529318},
      journal = {ACM Trans. Softw. Eng. Methodol.},
      month = {mar},
      keywords = {training programs, property-based debugging, neural networks}
    }
    
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
