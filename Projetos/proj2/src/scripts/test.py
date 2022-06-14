

from os.path import join

import sys
sys.path.append("/home/ricardonunes/Documents/VCOM/Projetos/proj2/src/")

import torch
import cnn
import loader
import utils

conf = utils.get_configs('proj.conf')

device = "cuda" if torch.cuda.is_available() else "cpu"

model = cnn.create_model(conf)
checkpoint = torch.load(join(utils.SRC_DIR, 'models/custom_955850_best_model.pth'))
model.load_state_dict(checkpoint['model'])
model = model.to(device)

test_dataloader = loader.get_test_loader(conf)

cnn.set_device(device)
loss_fn = cnn.get_loss_function(test_dataloader, conf['multilabel'])


test_loss, test_acc = cnn.epoch_iter(test_dataloader, model, loss_fn, is_train=False, multi_label=conf['multilabel'])
print(f"\nTest Loss: {test_loss:.3f} \nTest Accuracy: {test_acc:.3f}")

