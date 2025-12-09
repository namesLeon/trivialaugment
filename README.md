## Setup
Install a fitting PyTorch version for your setup. In my case, I used version 2.4.1+cu121 (RTX 4060 Laptop GPU).
Install the requirements found in requirements_updated.txt. Use python 3.8.

```
pip install -r requirements_updated.txt
# Install a pytorch version, in many setups this has to be done manually, see pytorch.org
```

Now you should be ready to go. Start a training like so:
```
python -m TrivialAugment.train -c confs/your_chosen_config.yaml --dataroot data --tag EXPERIMENT_NAME --save path/to/your/save/directory/save.pth
```

Adding the --save tag will lead do automatically saving checkpoints of your trained model every 20 epochs. In the event of an abrupt stop of the training process, the next training will resume from the last checkpoint.

## Configs and Logs

The configs used in the paper can be found in confs/wrn28x2 and in confs/wrn40x2.

For logs and metrics use a `tensorboard` with the `logs` directory.


## Trained models
Due to size constraints, I do not include the trained models in the codebase by 
default. You can find and download them at https://drive.google.com/drive/folders/1PleWGWxTTswplXtpfWsKf4wcAEzanJeM?usp=sharing.

## Confidence Intervals
You can find the code I used to compute the results and their 95% confidence 
intervals in results.py.

## Experiments
You can find the code I used to automatically do 10 consecutive experiments 
(usually overnight) in run_experiments.py.


