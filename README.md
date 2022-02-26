# Creative Adversarial Networks(CAN) Pytorch-Implementation

## About
This is an unofficial implementation of [CAN: Creative Adversarial Networks, Generating "Art" by Learning About Styles and Deviating from Style Norms](https://arxiv.org/abs/1706.07068) with Pytorch.

## Setting Parameters
Most of the major parameters used for learning are stored in `get_parameter.py`.
Please modify them as appropriate while learning.

```
def get_param():
    param_d = {
        'epochs': 500,
        'lr': 0.0001,
        'batch_size': 256,
        'img_size': 32,
        'channels': 3,
        'dataset_name': 'cifar10',
        'category_num': 10,
        'input_noise_size': 100,
        'ckp_path': 'CAN_ckp.pth'
    }
```

## Training CAN
```
python train.py
```
During training, the fake images produced by the CAN, the learning loss values and the weights of the model are stored.
They are saved as `visualize/fake_epoch{epoch}.png`, `training_loss.png`, and `ckp/net_G(or D)_CAN_ckp.pth`, respectively.

## Citation
```
@misc{2017cans,
  author = {Phillip Kravtsov and Phillip Kuznetsov},
  title = {Creative Adversarial Networks},
  year = {2017},
  howpublished = {\url{https://github.com/mlberkeley/Creative-Adversarial-Networks}},
  note = {commit xxxxxxx}
}
```
