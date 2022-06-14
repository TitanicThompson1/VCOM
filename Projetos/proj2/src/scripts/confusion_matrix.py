from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
import seaborn as sn
import pandas as pd
import torch.nn.functional as F

from os.path import join

import sys
sys.path.append("/home/ricardonunes/Documents/VCOM/Projetos/proj2/src/")

import utils
import loader
import cnn

y_pred = []
y_true = []

conf = utils.get_configs('proj.conf')

_, testloader = loader.get_loader(conf)
device = "cuda" if torch.cuda.is_available() else "cpu"

net = cnn.create_model(conf)
checkpoint = torch.load(join(utils.SRC_DIR, 'models/resnet_405054_latest_model.pth'))
net.load_state_dict(checkpoint['model'])

net.to(device)

# iterate over test data
for ind, (inputs, labels) in enumerate(testloader):
        inputs, labels = inputs.to(device), labels.to(device)
        output = net(inputs) # Feed Network

        probs = F.softmax(output, dim=1)
        final_pred = torch.argmax(probs, dim=1)
        fpred = final_pred[0].item() 
        y_pred.append(fpred) # Save Prediction
        
        label = labels[0].item()
        y_true.append(label) # Save Truth

# constant for classes
classes = ('trafficlight', 'stop', 'speedlimit', 'crosswalk')

# Build confusion matrix
cf_matrix = confusion_matrix(y_true, y_pred)

df_cm = pd.DataFrame(cf_matrix, index = [i for i in classes],
                     columns = [i for i in classes])
plt.figure(figsize = (12,7))
sn.heatmap(df_cm, annot=True)
plt.show()