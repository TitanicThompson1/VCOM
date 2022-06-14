from loader import get_test_loader
from utils import get_configs
import cnn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

config = get_configs('proj.conf')

test_dataloader = get_test_loader(config)

model = cnn.create_model(config).to('cuda')
model = model.eval()

num_classes = 2  # 1 class (person) + background

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
