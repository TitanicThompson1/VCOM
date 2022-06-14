import random
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, multilabel_confusion_matrix
import torch.nn.functional as F
import torch
import numpy as np
from tqdm import tqdm
from torchvision import models
from torch import nn
import utils
from sklearn.utils import class_weight
import custom_model
from os.path import join
import seaborn as sn


DEVICE = ''
TRAIN = 0
VAL = 1


def create_model(config):
  model_name = config['modelname']
  
  if model_name == 'resnet':
    resnet = models.resnet50(config['finetuning'])
    if config['finetuning']:
      freeze_model(resnet, config)
    resnet.fc = nn.Linear(2048, 4)
    return resnet

  elif model_name == 'custom':  # Custom Model
    return custom_model.CustomModel()
  
  elif model_name == 'frnn':  # Faster RCNN
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True, progress=False,num_classes=5)
    return model


def train_model(model, data_loaders, config):

  global DEVICE

  DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

  model.to(DEVICE)

  loss_fn = get_loss_function(data_loaders[TRAIN], config['multilabel'])

  optimizer = get_optimizer(model, config)

  resnet_train_history, resnet_val_history, model_path = train(model, data_loaders, 
                                                  loss_fn, optimizer, config)

  plotTrainingHistory(resnet_train_history, resnet_val_history)

  return model_path


# Adapted from teacher notebook
def train(model, data_loaders, loss_fn, optimizer, config):

  model_id = random.randrange(1, 1000000)

  train_history = {'loss': [], 'accuracy': []}
  val_history = {'loss': [], 'accuracy': []}

  best_val_loss = np.inf

  model_latest_path = 'models/' + config['modelname'] + "_" + str(model_id) + '_latest_model.pth'
  model_best_path = 'models/' + config['modelname'] + "_" + str(model_id) + '_best_model.pth'

  print("Start training...")
  
  for t in range(config['maxepochs']):
    print(f"\nEpoch {t+1}")
    train_loss, train_acc = epoch_iter(data_loaders[TRAIN], model, loss_fn, optimizer, multi_label=config['multilabel'])
    print(f"Train loss: {train_loss:.3f} \t Train acc: {train_acc:.3f}")

    val_loss, val_acc = epoch_iter(data_loaders[VAL], model, loss_fn, is_train=False, multi_label=config['multilabel'])
    print(f"Val loss: {val_loss:.3f} \t Val acc: {val_acc:.3f}")

    # save model when val loss improves
    if val_loss < best_val_loss:
      best_val_loss = val_loss
      save_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': t}
      torch.save(save_dict, join(utils.SRC_DIR, model_best_path))
      

    # save latest model
    save_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': t}
    torch.save(save_dict, join(utils.SRC_DIR, model_latest_path) )

    # save training history for plotting purposes
    train_history["loss"].append(train_loss)
    train_history["accuracy"].append(train_acc)

    val_history["loss"].append(val_loss)
    val_history["accuracy"].append(val_acc)
  
  model_id += 1

  print("Finished")
  return train_history, val_history, model_best_path


def epoch_iter(dataloader, model, loss_fn, optimizer=None, is_train=True, multi_label=False):
    if is_train:
      assert optimizer is not None, "When training, please provide an optimizer."
      
    num_batches = len(dataloader)

    if is_train:
      model.train() # put model in train mode
    else:
      model.eval()

    total_loss = 0.0
    preds = []
    labels = []

    with torch.set_grad_enabled(is_train):
      for _, (X, y) in enumerate(tqdm(dataloader)):
          X, y = X.to(DEVICE), y.to(DEVICE)

          # Compute prediction error
          pred = model(X)

          loss = loss_fn(pred, y)

          if is_train:
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

          # Save training metrics
          total_loss += loss.item() 
          
          final_pred = None
          if multi_label:
            probs = torch.sigmoid(pred)
            final_pred = probs.round().detach().clone()
          else:
            probs = F.softmax(pred, dim=1)
            final_pred = torch.argmax(probs, dim=1)
          
          preds.extend(final_pred.cpu().numpy())
          labels.extend(y.cpu().numpy())

    plot_cf_matrix(labels, preds, multi_label)
    return total_loss / num_batches, accuracy_score(labels, preds)


def freeze_model(model, config):
  if config['modelname'] == 'resnet' and config['finetuning']:
    for param in model.parameters():
      param.requires_grad = False



def get_loss_function(dataloader, multi_label):
  if multi_label:
    return nn.BCEWithLogitsLoss()
  
  labels = utils.get_all_labels(dataloader)

  class_weights=class_weight.compute_class_weight(class_weight='balanced',classes=np.unique(labels),y=labels.numpy())
  class_weights=torch.tensor(class_weights,dtype=torch.float).to(DEVICE)
  
  return nn.CrossEntropyLoss(weight=class_weights) 
  

def get_optimizer(model, conf):
  
  parameters = None
  if conf['finetuning'] and conf['modelname'] == 'resnet':
    parameters = model.fc.parameters()
  else:
    parameters = model.parameters()
  return torch.optim.Adam(parameters, lr=conf['learningrate'])
  

# plots to see the model training
def plotTrainingHistory(train_history, val_history):
    plt.subplot(2, 1, 1)
    plt.title('Cross Entropy Loss')
    plt.plot(train_history['loss'], label='train')
    plt.plot(val_history['loss'], label='val')
    plt.legend(loc='best')

    plt.subplot(2, 1, 2)
    plt.title('Classification Accuracy')
    plt.plot(train_history['accuracy'], label='train')
    plt.plot(val_history['accuracy'], label='val')

    plt.tight_layout()
    plt.legend(loc='best')
    plt.show()


def set_device(device):
  global DEVICE
  DEVICE = device


def plot_cf_matrix(labels, preds, multi_label):
  """ Plots the confusion matrix (or matrices, in case of multi-label problem)
  """
  classes = ('trafficlight', 'stop', 'speedlimit', 'crosswalk')

  if multi_label:
    cf_matrix = multilabel_confusion_matrix(labels, preds)
    fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    
    for axes, cfs_matrix, label in zip(ax.flatten(), cf_matrix, classes):
        print_confusion_matrix(cfs_matrix, axes, label, ["N", "Y"])
    
    fig.tight_layout()
    plt.show()
    return

  cf_matrix = confusion_matrix(labels, preds)
  df_cm = pd.DataFrame(cf_matrix, index = [i for i in classes],
                    columns = [i for i in classes])
  plt.figure(figsize = (12,7))
  sn.heatmap(df_cm, annot=True)
  plt.show()

def print_confusion_matrix(confusion_matrix, axes, class_label, class_names, fontsize=14):

  df_cm = pd.DataFrame(
      confusion_matrix, index=class_names, columns=class_names,
  )

  try:
      heatmap = sn.heatmap(df_cm, annot=True, fmt="d", cbar=False, ax=axes)
  except ValueError:
      raise ValueError("Confusion matrix values must be integers.")
  heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
  heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
  axes.set_ylabel('True label')
  axes.set_xlabel('Predicted label')
  axes.set_title("Confusion Matrix for the class - " + class_label)