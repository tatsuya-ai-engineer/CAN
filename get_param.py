
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
    return param_d

param = get_param()