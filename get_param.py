
def get_param():
    param_d = {
        'epochs': 20,
        'lr': 0.0001,
        'batch_size': 64,
        'img_size': 32,
        'channels': 3,
        'dataset_name': 'cifar10',
        'category_num': 10,
        'input_noise_size': 100
    }
    return param_d

param = get_param()