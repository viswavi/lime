#!/usr/bin/env python
# coding: utf-8

# In[62]:

from __future__ import print_function

import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)


# In[73]:


import contrastlime
from pytorch_lightning.core.lightning import LightningModule

import contrastlime.lime_text
import numpy as np
from biasinbios.BertBio import BertBio
from biasinbios.data import ClassificationDataset, MyCollator, ClassificationData
from torch.utils.data import DataLoader

import sklearn
import pytorch_lightning as pl 
import transformers 
import pickle as pkl
import torch
import math
from tqdm import tqdm 
from torch.nn import Softmax
from transformers import AutoTokenizer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

import sklearn.metrics


# In[64]:
class BertModel(LightningModule):
    def __init__(self, checkpoint_file, tokenizer, device='cuda'):
        super().__init__()
        self.model = BertBio.load_from_checkpoint(checkpoint_file).to(device=device)
        self.model.eval()
        self.softmax = Softmax(dim=1)
        self.tokenizer = tokenizer
        self.device_=device

    def predict_proba(self, input_texts):
        all_probs = []
        """
        could do batches but later
        """
        for input_text in input_texts:
            input_ids = torch.LongTensor(self.tokenizer(input_text)['input_ids']).to(self.device_)
            input_ids = input_ids.reshape(1,-1)
            attens = torch.LongTensor([1]*len(input_ids)).to(self.device_)
            attens = attens.reshape(1,-1)
            # dummy label
            label = torch.LongTensor([0]).to(self.device_)
            logits, _ = self.model((input_ids, attens, label))
            probs = self.softmax(logits)
            all_probs.append(probs.tolist())
        return np.asarray(all_probs).squeeze(axis=1)

    def do_forward(self, batch):
        bio, atten, labels = batch
        atten = atten.to(device=self.device_)
        bio = torch.LongTensor(bio).to(device=self.device_)
        labels = torch.LongTensor(labels).to(device=self.device_)
        logits_a, _ = self.model((bio, atten, labels))
        pred = torch.argmax(logits_a, dim=1).tolist()
        #preds_a.extend(pred_a)
        prob = self.softmax(logits_a)[0,labels].tolist()
        #probs_a.extend(prob_a)
        return pred, prob








DATA_PATH = '../biasinbios/data_bios/'
CHKPT_SCRUBBED = '../biasinbios/chkpts_scrubbed/epoch=4-step=4663.ckpt'
CHKPT_UNSCRUBBED = '../biasinbios/chkpts_unscrubbed/epoch=4-step=4663.ckpt'
BERT = 'bert-base-uncased'
NUM_WORKERS = 4
BATCH_SIZE = 16

# In[65]:


with open(os.path.join(DATA_PATH, 'unscrubbed_test.pkl'), 'rb') as f:
    test_data = pkl.load(f)

occupation2label = {}
label2occ = {}
for occ in set(test_data['occupation']):
    idxs = np.where(np.array(test_data['occupation']) == occ)[0]
    lbl = np.array(test_data['label'])[idxs][0]
    occupation2label[occ] = lbl
    label2occ[lbl] = occ
# In[66]:


#test_data.keys()
collator = MyCollator(BERT)

# In[67]:
print("Loading test data")
# dataset = ClassificationDataset(tokenizer=tokenizer,
#                                 data_path=DATA_PATH)
# dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE,
#                         shuffle=False, num_workers=NUM_WORKERS, collate_fn=collator)
dm = ClassificationData(basedir=DATA_PATH, tokenizer_name=BERT, batch_size=BATCH_SIZE)
#     for (tokens, tokens_mask, nt_idx_matrix, labels) in dm.train_dataloader():
#         print(torch.tensor(tokens_mask[0].tokens).shape)

# labels = test_data['label']
bios = list(test_data['bio'])
# tokenizer = AutoTokenizer.from_pretrained(BERT, do_lower_case=True)
# bios_indices = tokenizer(bios)['input_ids']
# attention_masks = [torch.LongTensor([1]*len(ind)).reshape(1,-1) for ind in bios_indices]


# In[68]:
DEVICE = 'cpu'
if torch.cuda.is_available():
    print("using gpu")
    DEVICE='cuda'

print("Loading models...")
# model_a = BertBio.load_from_checkpoint(CHKPT_UNSCRUBBED)
# model_b = BertBio.load_from_checkpoint(CHKPT_SCRUBBED)
# model_a = model_a.to(device=DEVICE)
# model_b = model_b.to(device=DEVICE)
#
# model_a.eval()
# model_b.eval()

model_a = BertModel(checkpoint_file=CHKPT_UNSCRUBBED, tokenizer=dm.tokenizer, device=DEVICE)
model_b = BertModel(checkpoint_file=CHKPT_SCRUBBED, tokenizer=dm.tokenizer, device=DEVICE)

# In[76]:



#for atten, bio, label in tqdm(zip(attention_masks, bios_indices, labels), total=len(labels)):
if not os.path.exists("predictions.pkl"):
    preds_a = []
    preds_b = []
    probs_a = []
    probs_b = []
    labels_all = []
    total_ = math.ceil(len(bios)/BATCH_SIZE)

    print("Evaluating on test dataset and saving prediction probabilities")

    for i_batch, batch in tqdm(enumerate(dm.test_dataloader()), total=total_):
        bio, atten, labels = batch
        labels_all.extend(labels.tolist())
        pred_a, prob_a = model_a.do_forward(batch)
        pred_b, prob_b = model_b.do_forward(batch)
        # atten = atten.to(device=DEVICE)
        # bio = torch.LongTensor(bio).to(device=DEVICE)
        # labels = torch.LongTensor(labels).to(device=DEVICE)
        # logits_a, _ = model_a((bio, atten, labels))
        # pred_a = torch.argmax(logits_a, dim=1).tolist()
        preds_a.extend(pred_a)
        # prob_a = softmax(logits_a)[0,labels].tolist()
        probs_a.extend(prob_a)


        # logits_b, _ = model_a((bio, atten, labels))
        #pred_b = torch.argmax(logits_b, dim=1)
        preds_b.extend(pred_b)
        #prob_b = softmax(logits_b)[0,labels].tolist()
        probs_b.extend(prob_b)
        if len(pred_a) != len(pred_b):
            print(len(pred_a), len(pred_b))
    with open('predictions.pkl', 'wb') as f:
        pkl.dump({'pred_scrubbed': preds_b, 'pred_unscrubbed': preds_a, \
                'prob_scrubbed': probs_b, 'prob_unscrubbed': probs_a, 'labels_all': labels_all}, f)

else:
    with open('predictions.pkl', 'rb') as f:
        data_dict = pkl.load(f)
    preds_a = data_dict['pred_unscrubbed']
    preds_b = data_dict['pred_scrubbed']

    probs_a = data_dict['prob_unscrubbed']
    probs_b = data_dict['prob_scrubbed']
    labels_all = data_dict['labels_all']


    # In[79]:

print("Model A accuracy (unscrubbed model) ", sklearn.metrics.accuracy_score(labels_all, preds_a))


# In[80]:


print("Model B accuracy (scrubbed model) ", sklearn.metrics.accuracy_score(labels_all, preds_b))


# In[82]:


disagrees = np.where((np.asarray(probs_a) <=.3) & (np.asarray(probs_b) >=.75))[0]
example_idx = disagrees[42]

from importlib import reload
reload(contrastlime.lime_text)
reload(contrastlime)
LimeTextExplainer = contrastlime.lime_text.LimeTextExplainer
class_lbls = sorted(np.unique(labels_all))
class_names = [label2occ[j] for j in class_lbls]



# In[35]:


# Single-model interpretation of classifier A (forest of 500 trees)
class_explainer = LimeTextExplainer(class_names=class_names, mode='classification')
class_exp = class_explainer.explain_instance(bios[example_idx],
                                         model_a.predict_proba, labels = (preds_a[example_idx], ),
                                         num_features=10)
print('Document id: %d' % example_idx)
print('Classifier A: probability =', model_a.predict_proba([bios[example_idx]])[0,labels_all[example_idx]])
print('True class: %s' % labels_all[example_idx])

class_exp.show_in_notebook()


# In[36]:


# Single-model interpretation of classifier B (forest of 2 trees)
class_explainer = LimeTextExplainer(class_names=class_names, mode='classification')
class_exp = class_explainer.explain_instance(bios[example_idx],
                                         model_b.predict_proba,
                                         num_features=10)
print('Document id: %d' % example_idx)
print('Classifier B: ', model_b.predict_proba([bios[example_idx]])[0,labels_all[example_idx]])
print('True class: %s' % labels_all[example_idx])

class_exp.as_list()
class_exp.show_in_notebook()


# In[39]:


# Comparison interpretation, in classification mode
# Examining why Classifier A chose label 0 (wrong), relative to Classifier B.
class_explainer = LimeTextExplainer(class_names=class_names, mode='classification')
class_exp = class_explainer.explain_instance_contrast(bios[example_idx],
                                         model_a.predict_proba,
                                         model_b.predict_proba,
                                         label_style="classification",
                                         num_features=10,
                                         label_to_examine=1,
                                         model_names=["Unscrubbed Bert", "Scrubbed Bert"])
print('Document id: %d' % example_idx)
print('Classifier A:', model_a.predict_proba([bios[example_idx]])[labels_all[example_idx]])
print('Classifier B:', model_b.predict_proba([bios[example_idx]])[labels_all[example_idx]])
print('True class: %s' % labels_all[example_idx])

class_exp.show_in_notebook()


# In[40]:


# Comparison interpretation, in classification mode
# Examining why Classifier A chose label 0 (wrong), relative to Classifier B.
reg_explainer = LimeTextExplainer(mode='regression')
reg_exp = reg_explainer.explain_instance_contrast(bios[example_idx],
                                         model_a.predict_proba,
                                         model_b.predict_proba,
                                         label_style="regression",
                                         num_features=10,
                                         label_to_examine=1,
                                         model_names=["Classifier A", "Classifier B"])
print('Document id: %d' % example_idx)
print('Classifier A:', model_a.predict_proba([bios[example_idx]])[labels_all[example_idx]])
print('Classifier B:', model_b.predict_proba([bios[example_idx]])[labels_all[example_idx]])
print('True class: %s' % labels_all[example_idx])

reg_exp.show_in_notebook()


# In[ ]:




