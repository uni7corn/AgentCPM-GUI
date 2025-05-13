# ARL

ARL (Another asynchronous Reinforcement Learning framework) allow us to train a vision-language model with minimal modification of the hugginface transformers Trainer.

Current Support Features:

- Load balance for task feed, (multiturn) completions gathering between different nodes.
- Aynchronous rollout before model updating.
- FSDPv2 Support.

## Installation

Run following commands:

```bash
conda create -n arl python=3.11
conda activate arl
pip install -e requirements.txt
```

Note: You need to install pytorch>=2.6 and the latest transformers to run with FSDPv2.


## How To Use

### 1. Modify the training scripts

The example script is `fsdp.sh`, you should change following args before runing the code:

- Set `RUN_NAME` to any name you like
- `source ~/miniconda3/bin/activate arl` this should be modify according to your installation.
- `model_name_or_path` should be the path to the model you want to train.
- `dataset_name` and `eval_dataset_name` should be the path of processed datasets.
- `NODES` and `NUM_PROCESSES` should be set according to your cluster status.

Make sure you have install `pdsh` to start training.


### (Optional) 2. Modify the Loading and Forwarding Behavior

You could modify the loading behavior of your model in the `grpo.py`.

Some models takes different keys when forwarding, you may should modify the `_get_per_token_logps` method for `AsyncRLGRPOTrainer` in the `trainer.arl` to support your models.

### 3. Run the script

```bash
bash fsdp.sh
```
You can view your wandb for details running, the checkpoint will be saved under `output` folder.