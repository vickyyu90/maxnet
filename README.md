# A novel explainable neural network for Alzheimer's disease diagnosis

PyTorch implementation for MAXNet, HAM and PCR. 


## Installation

- Linux or macOS with python ≥ 3.6
- PyTorch ≥ 1.6
- torchvision that matches the Pytorch installation. You can install them together at [pytorch.org](https://pytorch.org/) to make sure of this.
- tensorboard (needed for visualization): `pip install tensorboard`
- pandas
- cv2
- clinicadl (convert the MRIs in nii.gz format into tensor versions (using pytorch format)): https://github.com/aramis-lab/AD-ML.

## Data preparation
This is used to save MRI in nii.gz format into pytorch .pt format.
`clinicadl extract CAPS_DIRECTORY t1-linear roi`

## Quick Start 

To train a `MAXNet` on ADNI, run:

```bash
python -m torch.distributed.launch --nproc <num-of-gpus-to-use> --master_port 12345 main_train.py \
--caps_dir <caps-path> --tsv_path <tsv-path> --epochs <num-of-epochs> --resume <checkpoint-path>
```

To evaluate a pre-trained `MAXNet` on ADNI val, run:
```bash
python -m torch.distributed.launch --nproc <num-of-gpus-to-use> --master_port 12345 main_evaluate.py \
--caps_dir <caps-path> --tsv_path <tsv-path> --resume <checkpoint-path>
```

## Evaluation

To reproduce the results in Figs 6 - 9, run:
```bash
python -m torch.distributed.launch --nproc <num-of-gpus-to-use> --master_port 12345 main_evaluate.py \
--resume <checkpoint-path> --eval_sample '136S0300' --sample_session 'ses-M01'
```


## Requirements of Computer Hardware 

GPU>=16GB memory

RAM>=128GB memory

Note: our model can be deployed and run on the AWS cloud using NVIDIA GPU capable instances (including g4, p3 and p3dn instances).


## License
This repository is released under the [Apache 2.0 license].

## Citation
If you find this repository useful, please consider citation:
```
@inproceedings{yu2022explainable,
  title={A novel explainable neural network for Alzheimer\'s disease diagnosis},
  author={Yu, Lu and Xiang, Wei and Fang, Juan and Chen, Yi-Ping Phoebe and Zhu, Ruifeng},
  year={2022}
}
```

