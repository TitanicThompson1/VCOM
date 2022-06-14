from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
from os.path import join

import sys
sys.path.append("/home/ricardonunes/Documents/VCOM/Projetos/proj2/src/")

import utils
import loader
import cnn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EXAMPLE = 20

conf = utils.get_configs('proj.conf')
test_loader = loader.get_test_loader(conf)

model = cnn.create_model(conf)
checkpoint = torch.load(join(utils.SRC_DIR, 'models/resnet729086_best_model.pth'))
model.load_state_dict(checkpoint['model'])
model = model.to(DEVICE)

plt.figure(figsize=(150, 150))

for ind, (X, y) in enumerate(test_loader):
      if ind >= 80: break
      X, y = X.to(DEVICE), y.to(DEVICE)    
      pred = model(X)
      probs = F.softmax(pred, dim=1)
      final_pred = torch.argmax(probs, dim=1)


      plt.subplot(9, 9, ind + 1)
      plt.axis("off")
      plt.text(0, -1, y[0].item(), fontsize=14, color='green') # correct
      plt.text(58, -1, final_pred[0].item(), fontsize=14, color='red')  # predicted
      plt.imshow(X[0][0,:,:].cpu(), cmap='gray')
plt.show()