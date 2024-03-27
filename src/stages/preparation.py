import pandas as pd
from fastai.vision.all import ImageDataLoaders, Resize, pd


def get_im_load(config):

    df_labels = pd.read_csv(config['data']['train_target'],
                            sep='\t',
                            names=[config['data']['train_col_name'], config['data']['train_col_label']])

    name = config['data']['train_col_name']
    df_labels[name] = df_labels[name].transform(lambda x: (x + '.jpg'))

    idl_params = {'path': config['data']['train_mels'],
                  'fn_col': config['data']['train_col_name'],
                  'label_col': config['data']['train_col_label'],
                  'item_tfms': [Resize(192, method='squish')],
                  'seed': config['base']['seed'],
                  'valid_pct': config['data']['valid_pct'],
                  'shuffle': config['data']['shuffle_data']}

    if config['data']['if_all_images']:
        im_load = ImageDataLoaders.from_df(df_labels[:], idl_params)
    else:
        size = config['data']['size']
        im_load = ImageDataLoaders.from_df(df_labels[:size], **idl_params)

    return im_load
