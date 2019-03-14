# %matplotlib inline

from PIL import Image
from matplotlib.pyplot import imshow
from matplotlib.pyplot import imsave
from matplotlib.pyplot import savefig
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
from torch import topk
import numpy as np
import skimage.transform
import torch
import os
import imghdr

DIR_PATH = "data/validation"
HEAT_MAP_PATH = "results/heatmaps"

for subdir, dirs, files in os.walk(DIR_PATH):
    for file in files:
        image_path = os.path.join(subdir, file)
        if imghdr.what(image_path) == "jpeg":
            print("creating heatmaps for image %s of %s images: " % (files.index(file), len(files)))
            image = Image.open(image_path)
            imshow(image)

            # Imagenet mean/std

            normalize = transforms.Normalize(
               mean=[0.485, 0.456, 0.406],
               std=[0.229, 0.224, 0.225]
            )

            # Preprocessing - scale to 224x224 for model, convert to tensor,
            # and normalize to -1..1 with mean/std for ImageNet

            preprocess = transforms.Compose([
               transforms.Resize((224,224)),
               transforms.ToTensor(),
               normalize
            ])

            display_transform = transforms.Compose([
               transforms.Resize((224,224))])

            tensor = preprocess(image)
            prediction_var = Variable((tensor.unsqueeze(0)).cuda(), requires_grad=True)
            # model = models.resnet18(pretrained=True)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            model = models.resnet50(pretrained=False).to(device)
            model.cuda()
            model.eval()

            class SaveFeatures():
                features=None
                def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
                def hook_fn(self, module, input, output): self.features = ((output.cpu()).data).numpy()
                def remove(self): self.hook.remove()

            final_layer = model._modules.get('layer4')
            activated_features = SaveFeatures(final_layer)

            prediction = model(prediction_var)
            pred_probabilities = F.softmax(prediction).data.squeeze()
            activated_features.remove()

            topk(pred_probabilities,1)

            def getCAM(feature_conv, weight_fc, class_idx):
                _, nc, h, w = feature_conv.shape
                cam = weight_fc[class_idx].dot(feature_conv.reshape((nc, h*w)))
                cam = cam.reshape(h, w)
                cam = cam - np.min(cam)
                cam_img = cam / np.max(cam)
                return [cam_img]

            weight_softmax_params = list(model._modules.get('fc').parameters())
            weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())
            weight_softmax_params
            class_idx = topk(pred_probabilities,1)[1].int()
            overlay = getCAM(activated_features.features, weight_softmax, class_idx )
            imshow(overlay[0],alpha=0.5, cmap='jet')
            imshow(display_transform(image))
            imshow(skimage.transform.resize(overlay[0], tensor.shape[1:3]),alpha=0.5, cmap='jet');
            save_dir = os.path.join(HEAT_MAP_PATH, file)
            savefig(save_dir)
