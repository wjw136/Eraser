# command to evalute Eraser

## Installation
Install a python environment and install dependency:
```bash
conda env create -f environment.yaml
```

## Models
Download the [Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) and store it in ./models/Llama2-7b-chat-hf/

## Evaluting the defense capability:
AIM attack:
```bash
cd Evaluate_defense_capability
bash run_AIM.sh
```
AutoDAN attack:
```bash
bash run_AutoDAN.sh
```

## Evaluting the general capability:
```bash
cd Evaluate_general_capability
bash run.sh
```
