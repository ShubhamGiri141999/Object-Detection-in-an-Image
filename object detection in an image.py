#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torchvision
from pathlib import Path
from PIL import Image
import cv2
import matplotlib.pyplot as plt


# In[2]:


pip install opencv-python


# In[3]:


model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()


# In[4]:


COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


# In[5]:


def get_prediction(img_path, threshold):

    img = Image.open(img_path) 
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()]) 
    img = transform(img) 
    pred = model([img])

    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]

    pred_score = list(pred[0]['scores'].detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1] 
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    return pred_boxes, pred_class


# In[6]:


def object_detection_api(img_path, threshold=0.5, rect_th=3, text_size=3, text_th=3):
 
    boxes, pred_cls = get_prediction(img_path, threshold) 
    img = cv2.imread(img_path) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for i in range(len(boxes)):
        cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th)
        cv2.putText(img,pred_cls[i], boxes[i][0],  cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th) 

    plt.figure(figsize=(20,30)) 
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()


# In[7]:


object_detection_api('./s1.jpg', threshold=0.8)


# In[9]:


pip install --upgrade pip


# In[ ]:





# In[ ]:




