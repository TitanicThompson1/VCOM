from loader import get_loader
from utils import get_configs
import cnn

config = get_configs('proj.conf')

data_loaders = get_loader(config)

model = cnn.create_model(config)

path_to_model = cnn.train_model(model, data_loaders, config)