##  A Two-branch Neural Network for Non-homogeneous Dehazing via Ensemble Learning (https://arxiv.org/pdf/2104.08902.pdf)
### Dependencies and Installation

* python3.7
* PyTorch >= 1.0
* NVIDIA GPU+CUDA
* numpy
* matplotlib
* tensorboardX(optional)

### Pretrained Weights & Dataset

- Download [ImageNet pretrained weights](https://drive.google.com/file/d/1aZQyF16pziCxKlo7BvHHkrMwb8-RurO_/view?usp=sharing) and [our model weights](https://drive.google.com/file/d/1M2n6g7S5_sqPmTIAuI-IC30fhUmQr199/view?usp=sharing).
- Download our [dataset](https://drive.google.com/drive/folders/1eeBA2V_l9-evSJ0XWhRAww6ftweq8hU_?usp=sharing)


  
#### Train
```shell
python train.py --data_dir data -train_batch_size 8 --model_save_dir train_result
```

#### Test
 ```shell
python test.py --model_save_dir results
 ```



## Qualitative Results

Results on NTIRE 2021 NonHomogeneous Dehazing Challenge testing images:

<div style="text-align: center">
<img alt="" src="/images/test_results.png" style="display: inline-block;" />
</div>

## Citation

If you use any part of this code, please kindly cite

```
@article{
}
```



