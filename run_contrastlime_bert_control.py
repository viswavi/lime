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
import matplotlib.pyplot as plt
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
import argparse
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

        #preds_a.extend(pred_a)
        prob = (self.softmax(logits_a))[[i for i in range(len(bio))],labels].tolist()
        pred = torch.argmax(logits_a, dim=1).tolist()
        #probs_a.extend(prob_a)
        return pred, prob


parser = argparse.ArgumentParser()
parser.add_argument('--scrubbed', action='store_true', help='load scrubbed model and its control for comparison')

args = parser.parse_args()



DATA_PATH = '../biasinbios/data_bios/'
CHKPT_SCRUBBED = '../biasinbios/chkpts_scrubbed/epoch=4-step=4663.ckpt'
CHKPT_SCRUBBED_CONTROL = '../biasinbios/chkpts_scrubbed_4201/epoch=4-step=4663.ckpt'
CHKPT_UNSCRUBBED = '../biasinbios/chkpts_unscrubbed/epoch=4-step=4663.ckpt'
CHKPT_UNSCRUBBED_CONTROL = '../biasinbios/chkpts_unscrubbed_4201/epoch=4-step=4663.ckpt'
if args.scrubbed:
    print("settuping up control experiment for scrubbed model")
    MODELA = CHKPT_SCRUBBED
    MODELB = CHKPT_SCRUBBED_CONTROL
    scrub_arg = 'scrubbed'
else:
    print("settuping up control experiment for unscrubbed model")
    MODELA = CHKPT_UNSCRUBBED
    MODELB = CHKPT_UNSCRUBBED_CONTROL
    scrub_arg = 'unscrubbed'


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

np.random.seed(42)
model_a = BertModel(checkpoint_file=MODELA, tokenizer=dm.tokenizer, device=DEVICE)
model_b = BertModel(checkpoint_file=MODELB, tokenizer=dm.tokenizer, device=DEVICE)

# In[76]:
if not os.path.exists('lime_control_{}'.format(scrub_arg)):
    os.mkdir('lime_control_{}'.format(scrub_arg))
    os.mkdir('lime_control_{}'.format(scrub_arg) + '/class_exp')
    os.mkdir('lime_control_{}'.format(scrub_arg) + '/reg_exp_contrast')


#for atten, bio, label in tqdm(zip(attention_masks, bios_indices, labels), total=len(labels)):
if not os.path.exists('predictions_for_control_{}.pkl'.format(scrub_arg)):
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

        preds_a.extend(pred_a)
        probs_a.extend(prob_a)

        preds_b.extend(pred_b)

        probs_b.extend(prob_b)
        if len(pred_a) != len(pred_b):
            print(len(pred_a), len(pred_b))
    with open('predictions_for_control_{}.pkl'.format(scrub_arg), 'wb') as f:
        pkl.dump({'pred_{}_control'.format(scrub_arg): preds_b, 'pred_{}'.format(scrub_arg): preds_a, \
                'prob_{}_control'.format(scrub_arg): probs_b, 'prob_{}'.format(scrub_arg): probs_a, 'labels_all': labels_all}, f)

else:
    with open('predictions_for_control_{}.pkl'.format(scrub_arg), 'rb') as f:
        data_dict = pkl.load(f)
    preds_a = data_dict['pred_{}'.format(scrub_arg)]
    preds_b = data_dict['pred_{}_control'.format(scrub_arg)]

    probs_a = data_dict['prob_{}'.format(scrub_arg)]
    probs_b = data_dict['prob_{}_control'.format(scrub_arg)]
    labels_all = data_dict['labels_all']


    # In[79]:

print("Model A accuracy ({} model) ".format(scrub_arg), sklearn.metrics.accuracy_score(labels_all, preds_a))


# In[80]:


print("Model B accuracy ({} model, control) ".format(scrub_arg), sklearn.metrics.accuracy_score(labels_all, preds_b))


# In[82]:
from importlib import reload
reload(contrastlime.lime_text)
reload(contrastlime)
class_lbls = sorted(np.unique(labels_all))
class_names = [label2occ[j] for j in class_lbls]

disagrees = np.where((np.asarray(probs_a) <=.3) & (np.asarray(probs_b) >=.75))[0]
disagrees2 = np.where((np.asarray(probs_a) >=.75) & (np.asarray(probs_b) <=.3))[0]

additional1 = np.random.randint(0, len(disagrees), size=(1, 40))
additional2 = np.random.randint(0, len(disagrees2), size=(1, 40))


additional1 = set(additional1[0].tolist())
additional2 = set(additional2[0].tolist())

example_idxs = additional1 | set([42, 1200, 534, 89, 75, 84, 954, 342, 623, 777])
example_idxs2 = additional2 | set([4, 32, 987, 1000, 444, 324, 534, 212, 647, 999])
print ("--------------------")
print("looking through examples where B likely is wrong and A is likely correct")
print ("--------------------")
for example in example_idxs2:
    print ("--------------------")
    print ("--------------------")

    print(example)
    example_idx = disagrees2[example]


    LimeTextExplainer = contrastlime.lime_text.LimeTextExplainer
    class_lbls = sorted(np.unique(labels_all))
    class_names = [label2occ[j] for j in class_lbls]

    # # Single-model interpretation of classifier A
    class_explainer = LimeTextExplainer(class_names=class_names, mode='classification')
    class_exp = class_explainer.explain_instance(bios[example_idx],
                                                 model_a.predict_proba, labels = (labels_all[example_idx], ),
                                                 num_features=10)
    print ("--------------------")
    print ("Single Limes")
    print('Document id: %d' % example_idx)

    print('Classifier A {}: probability ='.format(scrub_arg), model_a.predict_proba([bios[example_idx]])[0,labels_all[example_idx]])
    print('True class: %s' % label2occ[labels_all[example_idx]])
    print('Classifier A predicted class: %s' % label2occ[preds_a[example_idx]])

    fig = class_exp.as_pyplot_figure(label=labels_all[example_idx])
    title = "Single LIME, {} Model, Predicted {}, True {}".format(scrub_arg, label2occ[preds_a[example_idx]], label2occ[labels_all[example_idx]])
    plt.title(title)
    fig.savefig('lime_control_{}/class_exp/example_{}_a_higher_a_{}.png'.format(scrub_arg, example_idx, label2occ[labels_all[example_idx]]))




    # Single-model interpretation of classifier B
    class_explainer = LimeTextExplainer(class_names=class_names, mode='classification')
    class_exp = class_explainer.explain_instance(bios[example_idx],
                                                 model_b.predict_proba,labels = (labels_all[example_idx], ),
                                                 num_features=10)


    print('Document id: %d' % example_idx)
    print('Classifier B {}, control: '.format(scrub_arg), model_b.predict_proba([bios[example_idx]])[0,labels_all[example_idx]])
    print('Classifier B predicted class: %s' % label2occ[preds_b[example_idx]])

    print('True class: %s' % labels_all[example_idx])

    class_exp.as_list(label=labels_all[example_idx])
    fig = class_exp.as_pyplot_figure(label=labels_all[example_idx])
    title = "Single LIME, {} Model control, Predicted {}, True {}".format(scrub_arg, label2occ[preds_b[example_idx]], label2occ[labels_all[example_idx]])
    plt.title(title)
    fig.savefig('lime_control_{}/class_exp/example_{}_b_higher_a_{}.png'.format(scrub_arg,example_idx, label2occ[labels_all[example_idx]]))
    print ("--------------------")



    print ("--------------------")
    print ("Contrastive Limes")
    reg_explainer = LimeTextExplainer(mode='regression')
    reg_exp = reg_explainer.explain_instance_contrast(bios[example_idx],
                                                      model_a.predict_proba,
                                                      model_b.predict_proba, num_samples=5000,
                                                      labels = (labels_all[example_idx], ),
                                                      num_features=10,
                                                      label_to_examine=labels_all[example_idx],
                                                      model_names=["Classifier A", "Classifier B"])
    print('Document id: %d' % example_idx)
    print('Classifier A {}:'.format(scrub_arg), model_a.predict_proba([bios[example_idx]])[0, labels_all[example_idx]])
    print('Classifier B {}, control:'.format(scrub_arg), model_b.predict_proba([bios[example_idx]])[0, labels_all[example_idx]])
    print('True class: %s' % label2occ[labels_all[example_idx]])

    fig = reg_exp.as_pyplot_figure(label=labels_all[example_idx])
    title = "Contrast LIME, {}, Model A Predicted {}, \n Model B Control Predicted {}, True {}".format(scrub_arg, label2occ[preds_a[example_idx]],\
                                                                                                           label2occ[preds_b[example_idx]], \
                                                                                                           label2occ[labels_all[example_idx]])
    plt.title(title)
    fig.savefig('lime_control_{}/reg_exp_contrast/example_{}_higher_a_{}.png'.format(scrub_arg, example_idx, label2occ[labels_all[example_idx]]))
    print ("--------------------")



print ("--------------------")
print("looking through examples where A likely is wrong and B is likely correct")
print ("--------------------")
for example in example_idxs:
    print ("--------------------")
    print ("--------------------")

    print(example)
    example_idx = disagrees[example]


    LimeTextExplainer = contrastlime.lime_text.LimeTextExplainer
    class_lbls = sorted(np.unique(labels_all))
    class_names = [label2occ[j] for j in class_lbls]

    # # Single-model interpretation of classifier A
    class_explainer = LimeTextExplainer(class_names=class_names, mode='classification')
    class_exp = class_explainer.explain_instance(bios[example_idx],
                                                 model_a.predict_proba, labels = (labels_all[example_idx], ),
                                                 num_features=10)
    print ("--------------------")
    print ("Single Limes")
    print('Document id: %d' % example_idx)

    print('Classifier A {}: probability ='.format(scrub_arg), model_a.predict_proba([bios[example_idx]])[0,labels_all[example_idx]])
    print('True class: %s' % label2occ[labels_all[example_idx]])
    print('Classifier A predicted class: %s' % label2occ[preds_a[example_idx]])

    fig = class_exp.as_pyplot_figure(label=labels_all[example_idx])
    title = "Single LIME, {} Model, Predicted {}, True {}".format(scrub_arg, label2occ[preds_a[example_idx]], label2occ[labels_all[example_idx]])
    plt.title(title)
    fig.savefig('lime_control_{}/class_exp/example_{}_a_higher_b_{}.png'.format(scrub_arg, example_idx, label2occ[labels_all[example_idx]]))




    # Single-model interpretation of classifier B
    class_explainer = LimeTextExplainer(class_names=class_names, mode='classification')
    class_exp = class_explainer.explain_instance(bios[example_idx],
                                                 model_b.predict_proba,labels = (labels_all[example_idx], ),
                                                 num_features=10)


    print('Document id: %d' % example_idx)
    print('Classifier B {}, control: '.format(scrub_arg), model_b.predict_proba([bios[example_idx]])[0,labels_all[example_idx]])
    print('Classifier B predicted class: %s' % label2occ[preds_b[example_idx]])

    print('True class: %s' % labels_all[example_idx])

    class_exp.as_list(label=labels_all[example_idx])
    fig = class_exp.as_pyplot_figure(label=labels_all[example_idx])
    title = "Single LIME, {} Model control, Predicted {}, True {}".format(scrub_arg, label2occ[preds_b[example_idx]], label2occ[labels_all[example_idx]])
    plt.title(title)
    fig.savefig('lime_control_{}/class_exp/example_{}_b_higher_b_{}.png'.format(scrub_arg,example_idx, label2occ[labels_all[example_idx]]))
    print ("--------------------")



    print ("--------------------")
    print ("Contrastive Limes")
    reg_explainer = LimeTextExplainer(mode='regression')
    reg_exp = reg_explainer.explain_instance_contrast(bios[example_idx],
                                                      model_a.predict_proba,
                                                      model_b.predict_proba, num_samples=5000,
                                                      labels = (labels_all[example_idx], ),
                                                      num_features=10,
                                                      label_to_examine=labels_all[example_idx],
                                                      model_names=["Classifier A", "Classifier B"])
    print('Document id: %d' % example_idx)
    print('Classifier A {}:'.format(scrub_arg), model_a.predict_proba([bios[example_idx]])[0, labels_all[example_idx]])
    print('Classifier B {}, control:'.format(scrub_arg), model_b.predict_proba([bios[example_idx]])[0, labels_all[example_idx]])
    print('True class: %s' % label2occ[labels_all[example_idx]])

    fig = reg_exp.as_pyplot_figure(label=labels_all[example_idx])
    title = "Contrast LIME, {}, Model A Predicted {}, \n Model B Control Predicted {}, True {}".format(scrub_arg, label2occ[preds_a[example_idx]], \
                                                                                                       label2occ[preds_b[example_idx]], \
                                                                                                       label2occ[labels_all[example_idx]])
    plt.title(title)
    fig.savefig('lime_control_{}/reg_exp_contrast/example_{}_higher_b_{}.png'.format(scrub_arg, example_idx, label2occ[labels_all[example_idx]]))
    print ("--------------------")

