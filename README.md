# DebiasingConGater

This repository is dedicated to the models of the paper [Controllable Gate Adapters (ConGater)](https://arxiv.org/abs/2401.16457) Accepted to main conference of EACL 2024.

The infrastructure of the code is similar to the [ModularizedDebiasing](https://github.com/CPJKU/ModularizedDebiasing) from the papers:

[Modular and On-demand Bias Mitigation with Attribute-Removal Subnetworks](https://aclanthology.org/2023.findings-acl.386.pdf)

and 

[Parameter-efficient Modularised Bias Mitigation via AdapterFusion](https://aclanthology.org/2023.eacl-main.201.pdf)


# Tasks and Environments

## Classification Task 

### Environment 
To Install the environment after instilling Conda or Mini conda run the command 

```
conda env create -f cls_congater.yml
```

After installation is complete you can access the environment by running :

```
conda activate cls_congater
```

### Structure of the code 

* **scripts**: Folder contains utility codes for adversarial mode, checkpoint, generating embeddings and etc. 

* **src**: Folder contains utility codes for attacking , evaluating , logging and data handler

* **models**: Subfolder of src contains all the models that can be called for training which contain , Baseline, Baseline adv, Adapter, Adapter Adv and ConGater 

* **cfg.yml**: contains all the training and dataset configurations required to run the training with default values

* **main_attack.py**: can be used to attack an already trained model (normally called with main.py)

* **main.py**: contains train wrappers and arguments to overwrite default config file. 

### Changing Code using Arguments:

To run the code you can use the following python command.

```
python main.py --gpu_id=0 --ds=hatespeech --model_type=congater --training_method=par --model_name=mini --random_seed --num_runs=1 --gate_squeeze_ratio=12 --log_wandb
```

* **--gpu_id**: sets the id of the gput that you are using by default 0 
* **--ds**: sets the dataset that you want to run the model on 
* **--model_type**: sets the model type that you want to run (baseline, baseline_adv,adapter_baseline, adapter_adv, congater)
* **--training_method**: For ConGater you can select parallel (par) training or post-hoc(post) training 
* **--model_name**: sets the model to run (bert,mini,roberta-base) are the values 
* **--random_seed**: activates random seeding for several runs to insure we dont have seed selection bias
* **--num_runs**: select how many times you want the current cofig to run , (always use with --random_seed)
* **--gate_squeeze_ratio**: selects the bottleneck for the ConGater , for adapter the default values are in the cofig based on previous papers
* **--log_wandb**: Starts logging the runs on wandb 
* **--wandb_project**: sets the prefix of the project, dataset name is added to the project name by default

For More control arguments please check out main.py file line 543-580.


# Paper citation:
```
@inproceedings{masoudian2024congater,
	title        = {Effective Controllable Bias Mitigation for Classification and Retrieval using Gate Adapters},
	author       = {Shahed Masoudian and Cornelia Volaucnik and Markus Schedl and Navid Rekabsaz},
	year         = 2024,
	booktitle    = {Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics},
	publisher    = {Association for Computational Linguistics},
	address      = {Malta}
}
```


# Information Retrieval Task




