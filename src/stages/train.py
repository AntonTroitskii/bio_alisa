import os
from torchvision import transforms, datasets

from mllib.utils.pkl import save_data_as_pkl
from mllib.utils.io import make_folder
from mllib.pytorch.files import save_results_as_csv
from pathlib import Path
from torch.utils.data import DataLoader
from torch import nn
from mllib.utils.dvc import load_params
from torch.utils.data import Subset
import torch
from torch.utils.data import random_split
from mllib.pytorch.train import train
from src.train.models import get_torch_model
from mllib.pytorch.plot import save_plot_results
from mllib.utils.logs import get_logger


def train_model():
    params = load_params()
    logger = get_logger("TRAIN", log_level=params['base']['log_level'])

    train_path = Path(params['data']['train_mels'])
    model_path = Path(params['train']['model'])
    n_images = params['train']['number_images']

    resluts_path_csv = Path(params['results']['train_results_csv'])
    resluts_path_png = Path(params['results']['train_results_png'])

    image_transform = transforms.Compose(
        [transforms.Resize(size=(64, 64)), transforms.ToTensor()]
    )

    train_dataset = datasets.ImageFolder(root=train_path,
                                         transform=image_transform,
                                         target_transform=None)

    if n_images:
        train_dataset = Subset(train_dataset, torch.arange(0, n_images))
        logger.info('Number images to process {}'.format(n_images))

    # Split data

    train_data, test_data = random_split(train_dataset, lengths=[
                                         0.8, 0.2], generator=torch.Generator().manual_seed(params['base']['seed']))

    logger.info('Train data size {}, test data size {}'.format(
        len(train_data), len(test_data)))

    # Create DataLoaders
    train_dataloader = DataLoader(
        dataset=train_data, batch_size=24, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(
        dataset=test_data, batch_size=24, shuffle=12, num_workers=0)

    model_name = params['train']['nn']
    model_weight = params['train']['weights']
    lr = params['train']['lr']
    epochs = params['train']['epochs']

    model = get_torch_model(model_name=model_name, model_weight=model_weight)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger.info('TRAIN STARTS ...')
    results = train(model=model,
                    train_dataloader=train_dataloader,
                    test_dataloader=test_dataloader,
                    epochs=epochs, loss_fn=loss_fn,
                    optimizer=optimizer, device=device)
    logger.info('TRAIN COMPLEATES.')

    logger.info('Current directory {}'.format(os.getcwd()))
    make_folder(model_path.parent)
    save_data_as_pkl(model, model_path)

    make_folder(resluts_path_csv.parent)
    save_results_as_csv(results=results, file_path=resluts_path_csv)
    save_plot_results(results=results, file_path=resluts_path_png)

    return results


if __name__ == '__main__':
    # arg_parser = argparse.ArgumentParser()
    # arg_parser.add_argument('--config', dest='config', required=True)
    # args = arg_parser.parse_args()

    # train_model(config_path=args.config)

    train_model()
