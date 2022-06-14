import loader
from torchvision import transforms

data_transforms = {
        'train':
        transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]),
        'test':
        transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]), 
    }

dataset = loader.TrafficSignDataset(
    "proj2/src/dataset/train/", 
    "proj2/src/dataset/ann_train/", 
    transform=data_transforms['train'],
    multi_label=True)

n_stop = 0
n_trafficlight = 0
n_speedlimit = 0
n_crosswalk = 0

for img, label in dataset:
    if label[0] == 1:
        n_trafficlight += 1
    if label[1] == 1:
        n_stop += 1
    if label[2] == 1:
        n_speedlimit += 1
    if label[3] == 1:
        n_crosswalk += 1

print("Trafficlight: {}".format(n_trafficlight))
print("Stop: {}".format(n_stop))
print("Speedlimit: {}".format(n_speedlimit))
print("Crosswalk: {}".format(n_crosswalk))

print("Total: {}".format(len(dataset)))
print("Total counted: {}".format(n_trafficlight + n_stop + n_speedlimit + n_crosswalk))