'''
Based on https://github.com/microsoft/vscode/issues/125993 
only use known skin type
train dissentangle network
python -u train_DisCo.py 20 full fitzpatrick DisCo
python -u train_DisCo.py 15 full ddi DisCo
'''
from __future__ import print_function, division
from sklearn.decomposition import TruncatedSVD
import torch
from torchvision import transforms, models
import pandas as pd
import numpy as np
import os
import skimage
from skimage import io
import warnings
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import WeightedRandomSampler
from torch.optim import lr_scheduler
import time
import copy
import random
import sys
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import balanced_accuracy_score
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
# get model
from Models.got_losses import Network, Confusion_Loss, Supervised_Contrastive_Loss, BinaryMatrixGenerator
from Masked_GOT_NewSinkhorn import cost_matrix_batch_torch, GW_distance_uniform, IPOT_distance_torch_batch_uniform
from transformers import AutoTokenizer, AutoModel
#torch.autograd.set_detect_anomaly(True) # GHC added
# Reproducibility

warnings.filterwarnings("ignore")

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
bert_model = AutoModel.from_pretrained('bert-base-uncased')
bert_model.to(torch.device('cuda'))

labels_top = ['drug induced pigmentary changes', 'photodermatoses',
       'dermatofibroma', 'psoriasis', 'kaposi sarcoma',
       'neutrophilic dermatoses', 'granuloma annulare',
       'nematode infection', 'allergic contact dermatitis',
       'necrobiosis lipoidica', 'hidradenitis', 'melanoma',
       'acne vulgaris', 'sarcoidosis', 'xeroderma pigmentosum',
       'actinic keratosis', 'scleroderma', 'syringoma', 'folliculitis',
       'pityriasis lichenoides chronica', 'porphyria',
       'dyshidrotic eczema', 'seborrheic dermatitis', 'prurigo nodularis',
       'acne', 'neurofibromatosis', 'eczema', 'pediculosis lids',
       'basal cell carcinoma', 'pityriasis rubra pilaris',
       'pityriasis rosea', 'livedo reticularis',
       'stevens johnson syndrome', 'erythema multiforme',
       'acrodermatitis enteropathica', 'epidermolysis bullosa',
       'dermatomyositis', 'urticaria', 'basal cell carcinoma morpheiform',
       'vitiligo', 'erythema nodosum', 'lupus erythematosus',
       'lichen planus', 'sun damaged skin', 'drug eruption', 'scabies',
       'cheilitis', 'urticaria pigmentosa', 'behcets disease',
       'nevocytic nevus', 'mycosis fungoides',
       'superficial spreading melanoma ssm', 'porokeratosis of mibelli',
       'juvenile xanthogranuloma', 'milia', 'granuloma pyogenic',
       'papilomatosis confluentes and reticulate',
       'neurotic excoriations', 'epidermal nevus', 'naevus comedonicus',
       'erythema annulare centrifigum', 'pilar cyst',
       'pustular psoriasis', 'ichthyosis vulgaris', 'lyme disease',
       'striae', 'rhinophyma', 'calcinosis cutis', 'stasis edema',
       'neurodermatitis', 'congenital nevus', 'squamous cell carcinoma',
       'mucinosis', 'keratosis pilaris', 'keloid', 'tuberous sclerosis',
       'acquired autoimmune bullous diseaseherpes gestationis',
       'fixed eruptions', 'lentigo maligna', 'lichen simplex',
       'dariers disease', 'lymphangioma', 'pilomatricoma',
       'lupus subacute', 'perioral dermatitis',
       'disseminated actinic porokeratosis', 'erythema elevatum diutinum',
       'halo nevus', 'aplasia cutis', 'incontinentia pigmenti',
       'tick bite', 'fordyce spots', 'telangiectases',
       'solid cystic basal cell carcinoma', 'paronychia', 'becker nevus',
       'pyogenic granuloma', 'langerhans cell histiocytosis',
       'port wine stain', 'malignant melanoma', 'factitial dermatitis',
       'xanthomas', 'nevus sebaceous of jadassohn',
       'hailey hailey disease', 'scleromyxedema', 'porokeratosis actinic',
       'rosacea', 'acanthosis nigricans', 'myiasis',
       'seborrheic keratosis', 'mucous cyst', 'lichen amyloidosis',
       'ehlers danlos syndrome', 'tungiasis', 'eudermic']

def calculate_probabilities(string_list):
    n = len(string_list)
    counts = {}
    probabilities = []
    
    # Count occurrences of each string
    for string in string_list:
        counts[string] = counts.get(string, 0) + 1

    # Calculate probabilities
    summation = 0
    for string in string_list:
        probability = counts[string] / n
        probabilities.append(probability)
        summation+=probability
    probabilities = [i/summation for i in probabilities]

    return probabilities

def got_loss(p,q, Mask, lamb):
    #print('NAN values in p:', torch.sum(torch.isnan(p)).item())
    cos_distance = cost_matrix_batch_torch(p.transpose(2,1), q.transpose(2,1)).transpose(1,2)
    
    beta=0.8
    min_score = cos_distance.min()
    max_score = cos_distance.max()
    threshold = min_score + beta * (max_score - min_score)
    cos_dist = torch.nn.functional.relu(cos_distance - threshold)
    wd, T = IPOT_distance_torch_batch_uniform(cos_dist, Mask, p.size(0),p.size(1),q.size(1),30)
    gwd = GW_distance_uniform(p.transpose(2,1), q.transpose(2,1), Mask)
    twd = lamb * torch.mean(gwd) + (1 - lamb) * torch.mean(wd)
    return twd


def flatten(list_of_lists):
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list_of_lists[:1] + flatten(list_of_lists[1:])

def find_largest_parameter(model): # GHC Added
    largest_param = None
    largest_value = float('-inf')

    for name, param in model.named_parameters():
        if param.numel() > 0:  # Check if the parameter has elements
            param_max = torch.max(param)
            if param_max > largest_value:
                largest_param = name
                largest_value = param_max.item()

    return largest_param, largest_value

def train_model(label, dataloaders, device, dataset_sizes, model,
                criterion, optimizer, scheduler, num_epochs=2, alpha=1.0, beta=0.8):
    print('hyper-parameters alpha: {}  beta: {}'.format(alpha, beta))
    since = time.time()
    training_results = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 0.0
    train_step = 0 # for tensorboard
    leading_epoch = 0  # record best model epoch
    
    #partition = batch["partition"] + ['eudermic']
    # bert_model.to(device)
    # encoding = tokenizer(labels_top, padding='max_length', truncation=True, max_length=2, return_tensors='pt')
    # input_ids =  torch.tensor(encoding['input_ids'].squeeze()).to(device)
    # attention_mask =  torch.tensor(encoding['attention_mask'].squeeze()).to(device)
    # bert_output = bert_model(input_ids, attention_mask) # 16,17,768
    #
    #textual_embeddings = torch.cat((bert_output.last_hidden_state.view(114*2,768),)*32).view(32,114*2,768)
    #textual_embeddings_9 = torch.cat((bert_output.last_hidden_state.view(114*2,768),)*9).view(9,114*2,768)
    # emb = []
    # for label_ in labels_top:
    #     encoding = tokenizer(label_, return_tensors='pt')

    #     input_ids =  torch.tensor(encoding['input_ids']).to(device)

    #     bert_output = bert_model(input_ids)[1]
    #     emb.append(bert_output)

    # textual_embeddings = torch.cat(tuple([torch.tensor((torch.cat(tuple(emb)))).unsqueeze(0)]*32))
    #print(np.shape(textual_embeddings), 'TEXT EMBEDDINGS')
    text_embeddings = np.load('./text_embeddings_3_large_consecutive_averaged.npy')
    text_embeddings = np.array(text_embeddings, dtype=np.double)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                #scheduler.step()
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0
            running_balanced_acc_sum = 0.0
            # running_total = 0
            print(phase)
            # Iterate over data.
            loop = tqdm(dataloaders[phase], leave=True, desc=f" {phase}-ing Epoch {epoch + 1}/{num_epochs}")
            for n_iter, batch in enumerate(loop):
                bs = len(batch['mid'])
                textual_embeddings = torch.cat(tuple([torch.tensor(text_embeddings).unsqueeze(0)]*bs)).cuda().double()
                inputs = batch["image"].to(device)
                # note!!! skin type starts from 1, so subtract 1
                label_c, label_t = batch[label], batch['fitzpatrick']-1 # label_condition, label_type 
                # weights = batch['weight'].to(device)
                label_c, label_t = torch.from_numpy(np.asarray(label_c)).to(device), torch.from_numpy(np.asarray(label_t)).to(device)
                # zero the parameter gradients
                # x = calculate_probabilities(partition)
                # print(x, partition)
                # exit()
                
                # print('\n bert output last hidden state shape: ',np.shape(bert_output.last_hidden_state))
                # #for i in range(16):
                # bert_output.last_hidden_state = torch.cat((bert_output.last_hidden_state,torch.tensor([bert_output.last_hidden_state[-1]])*16),dim=0)

                # print(np.shape(bert_output.last_hidden_state[i]))
                # bert_output = [i[-1]+bert_output.last_hidden_state[-1] for i in bert_output.last_hidden_state][:-2]
                # print(np.shape(bert_output))
                # exit()
                # partition = batch["partition"] + ['eudermic']
                # bert_model.to(device)
                # encoding = tokenizer(partition, padding='max_length', truncation=True, max_length=2, return_tensors='pt')
                # input_ids =  torch.tensor(encoding['input_ids'].squeeze()).to(device)
                # attention_mask =  torch.tensor(encoding['attention_mask'].squeeze()).to(device)
                # bert_output = bert_model(input_ids, attention_mask) # 16,17,768
                # textual_embeddings = torch.cat((bert_output.last_hidden_state.view(66,768),)*32).view(32,66,768)
    
                #bert_output = y
                
                optimizer.zero_grad()
                
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    #print(scheduler.get_last_lr()[0])
                    inputs = inputs.float()  # ADDED AS A FIX
                    #print(inputs.shape)
                    output = model(inputs)
                    
                    l_got = got_loss(output[-1], textual_embeddings, output[3], lamb = 0.9)
                    
                    _, preds = torch.max(output[0], 1)  # branch 1 get condition prediction
                    loss0 = criterion[0](output[0], label_c) 
                    loss1 = criterion[1](output[1], label_t)  # branch 2 confusion 
                    loss2 = criterion[2](output[2], label_t)  # branch 2 ce loss
                    loss3 = torch.tensor(0)#criterion[3](output[3], label_c)  # supervised contrastive loss
                    loss = loss0 + loss1*0.5 + loss2  + 1*l_got #+loss3*beta
                    # backward + optimize only if in training phase
                    #print('Losses:', l_got.item(), loss0.item(), loss1.item(), loss2.item())
                    if phase == 'train':
                        loss.backward(retain_graph=True)
                        
                        
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Set max_norm as desired

                        #print('output grads',output[-1].grad)
                        optimizer.step()
                    # statistics
                # tensorboard
                if phase == 'train':
                    writer.add_scalar('Loss/'+phase, loss.item(), train_step)
                    writer.add_scalar('Loss/'+phase+'loss0', loss0.item(), train_step)
                    writer.add_scalar('Loss/'+phase+'loss1_conf', loss1.item(), train_step)
                    writer.add_scalar('Loss/'+phase+'loss2', loss2.item(), train_step)
                    writer.add_scalar('Loss/'+phase+'contrast_loss', loss3.item(), train_step)
                    writer.add_scalar('Accuracy/'+phase, (torch.sum(preds == label_c.data)).item()/inputs.size(0), train_step)
                    writer.add_scalar('Balanced-Accuracy/'+phase, balanced_accuracy_score(label_c.data.cpu(), preds.cpu()), train_step)
                    train_step += 1
                # -------------------------
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == label_c.data)
                running_balanced_acc_sum += balanced_accuracy_score(label_c.data.cpu(), preds.cpu())*inputs.size(0)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            epoch_balanced_acc = running_balanced_acc_sum / dataset_sizes[phase]
            # print("Loss: {}/{}".format(running_loss, dataset_sizes[phase]))
            print("Accuracy: {}/{}".format(running_corrects,
                                           dataset_sizes[phase]))
            print('{} Loss: {:.4f} Acc: {:.4f} Balanced-Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, epoch_balanced_acc))
            # tensorboard 
            writer.add_scalar('lr/'+phase, scheduler.get_last_lr()[0], epoch)
            if phase == 'val':
                writer.add_scalar('Loss/'+phase, epoch_loss, epoch)
                writer.add_scalar('Accuracy/'+phase, epoch_acc, epoch)
                writer.add_scalar('Balanced-Accuracy/'+phase, epoch_balanced_acc, epoch)
            # ---------------------    
            training_results.append([phase, epoch, epoch_loss, epoch_acc.item(), epoch_balanced_acc])
            if epoch > 0:
                if phase == 'val' and epoch_acc > best_acc:
                    print("New leading accuracy: {}".format(epoch_acc))
                    best_acc = epoch_acc
                    leading_epoch = epoch
                    best_model_wts = copy.deepcopy(model.state_dict())
                # use balanced acc
                # if phase == 'val' and epoch_balanced_acc > best_acc:
                #     print("New leading balanced accuracy: {}".format(epoch_balanced_acc))
                #     best_acc = epoch_balanced_acc
                #     leading_epoch = epoch
                #     best_model_wts = copy.deepcopy(model.state_dict())              
            elif phase == 'val':
                best_acc = epoch_acc
                # use balanced acc
                # best_acc = epoch_balanced_acc
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Best model epoch:', leading_epoch)
    model.load_state_dict(best_model_wts)
    training_results = pd.DataFrame(training_results)
    training_results.columns = ["phase", "epoch", "loss", "accuracy", "balanced-accuracy"]
    return model, training_results


class SkinDataset():
    def __init__(self, dataset_name, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.dataset_name == 'ddi':
            img_name = os.path.join(self.root_dir,
                                str(self.df.loc[self.df.index[idx], 'hasher']))
            image = cv2.imread(img_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            img_name = os.path.join(self.root_dir,
                                str(self.df.loc[self.df.index[idx], 'hasher']))+'.jpg'
            image = io.imread(img_name)

        if(len(image.shape) < 3):
            image = skimage.color.gray2rgb(image)

        hasher = self.df.loc[self.df.index[idx], 'hasher']
        high = self.df.loc[self.df.index[idx], 'high']
        # mid = self.df.loc[self.df.index[idx], 'mid']
        low = self.df.loc[self.df.index[idx], 'low']
        fitzpatrick = self.df.loc[self.df.index[idx], 'fitzpatrick']

        if self.dataset_name == 'fitzpatrick':
            mid = self.df.loc[self.df.index[idx], 'mid'] 
        else:
            mid = 0
        if self.dataset_name == 'fitzpatrick':
            mid = self.df.loc[self.df.index[idx], 'mid'] 
            partition = self.df.loc[self.df.index[idx], 'label'] 
        else:
            mid = 0
            partition = self.df.loc[self.df.index[idx], 'disease'] 

        if self.transform:
            image = self.transform(image)
        sample = {
                    'image': image,
                    'high': high,
                    'mid': mid,
                    'low': low,
                    'hasher': hasher,
                    'fitzpatrick': fitzpatrick,
                    "partition": partition
                }
        return sample


def custom_load(
        batch_size=32,
        num_workers=10,
        train_dir='',
        val_dir='',
        label = 'low',
        dataset_name = 'fitzpatrick',
        image_dir='~/Documents/MultiModal_Research/FairDisCo/fitzpatrick17k/data/finalfitz17k',
        ):
    if dataset_name == 'ddi':
        image_dir = 'images'
    val = pd.read_csv(val_dir)
    train = pd.read_csv(train_dir)
    class_sample_count = np.array(train[label].value_counts().sort_index())
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in train[label]])

    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(
        samples_weight.type('torch.DoubleTensor'),
        len(samples_weight),
        replacement=True)
    dataset_sizes = {"train": train.shape[0], "val": val.shape[0]}
    transformed_train = SkinDataset(
        dataset_name = dataset_name,
        csv_file=train_dir,
        root_dir=image_dir,
        transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=224),  # Image net standards
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
            ])
        )
    transformed_test = SkinDataset(
        dataset_name = dataset_name,
        csv_file=val_dir,
        root_dir=image_dir,
        transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        )
    dataloaders = {
        "train": torch.utils.data.DataLoader(
            transformed_train,
            batch_size=batch_size,
            sampler=sampler,
            # drop_last = True,
            #shuffle=True,
            num_workers=num_workers),
        "val": torch.utils.data.DataLoader(
            transformed_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers)
        }
    return dataloaders, dataset_sizes

if __name__ == '__main__':
    # In the custom_load() function, make sure to specify the path to the images
    print("\nPlease specify number of epochs and 'dev' mode or not... e.g. python train.py 10 full \n")
    n_epochs = int(sys.argv[1])
    dev_mode = sys.argv[2]
    dataset_name = sys.argv[3]
    model_name = sys.argv[4]
    domain = int(sys.argv[5]) # Domain

    torch.manual_seed(2)
    random.seed(2)
    np.random.seed(2)

    print("CUDA is available: {} \n".format(torch.cuda.is_available()))
    print("Starting... \n")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if dev_mode == "dev":
        if dataset_name == 'ddi':
            df = pd.read_csv('ddi_metadata_code.csv').sample(300)
        else:
            df = pd.read_csv("fitzpatrick17k_known_code.csv").sample(1000)
    else:
        if dataset_name == 'ddi':
            df = pd.read_csv('ddi_metadata_code.csv')
        else:
            df = pd.read_csv("fitzpatrick17k_known_code.csv")

    domain = ["random_holdout", "a12", "a34","a56"][domain]
    print("DOMAIN:", domain)
    for holdout_set in [domain]: # ["expert_select","random_holdout", "a12", "a34","a56", "dermaamin","br"]:
        if holdout_set == "expert_select":
            df2 = df
            train = df2[df2.qc.isnull()]
            test = df2[df2.qc=="1 Diagnostic"]  
        elif holdout_set == "random_holdout":
            if dataset_name == 'ddi':
                train, test, y_train, y_test = train_test_split(
                                                    df,
                                                    df['high'],
                                                    test_size=0.2,
                                                    random_state=2,
                )
            else:
                train, test, y_train, y_test = train_test_split(
                                                    df,
                                                    df['low'],
                                                    test_size=0.2,
                                                    random_state=2,
                                                    stratify=df['low']) # 
        elif holdout_set == "dermaamin": # train with b
            # only choose those skin conditions in both dermaamin and non dermaamin
            combo = set(df[df.url.str.contains("dermaamin")==True].label.unique()) & set(df[df.url.str.contains("dermaamin")==False].label.unique())
            count_atla = (df.loc[df.url.str.contains("dermaamin")==False]).label.value_counts()
            count_atla = count_atla.rename_axis('unique_values').reset_index(name='counts')
            combo = combo & set((count_atla.loc[count_atla['counts']>=5])['unique_values'])
            df = df[df.label.isin(combo)]
            # remove the class only has one sample
            df["low"] = df['label'].astype('category').cat.codes
            # train = df[df.image_path.str.contains("dermaamin") == False]
            # test = df[df.image_path.str.contains("dermaamin")]
            train_test = df[df.url.str.contains("dermaamin") == False]
            train, test, y_train, y_test = train_test_split(
                                                train_test,
                                                train_test['low'],
                                                test_size=0.2,
                                                random_state=2,
                                                stratify=train_test['low']) # 
            print(train['low'].nunique())
            print(test['low'].nunique())
            test2 = df[df.url.str.contains("dermaamin") == True]
        elif holdout_set == "br": # train with a
            # only choose those skin conditions in both dermaamin and non dermaamin
            combo = set(df[df.url.str.contains("dermaamin")==True].label.unique()) & set(df[df.url.str.contains("dermaamin")==False].label.unique())
            count_derm = (df.loc[df.url.str.contains("dermaamin")==True]).label.value_counts()
            count_derm = count_derm.rename_axis('unique_values').reset_index(name='counts')
            combo = combo & set((count_derm.loc[count_derm['counts']>=5])['unique_values'])
            df = df[df.label.isin(combo)]
            df["low"] = df['label'].astype('category').cat.codes
            # train = df[df.image_path.str.contains("dermaamin")]
            # test = df[df.image_path.str.contains("dermaamin") == False]
            train_test = df[df.url.str.contains("dermaamin") == True]
            train, test, y_train, y_test = train_test_split(
                                                train_test,
                                                train_test['low'],
                                                test_size=0.2,
                                                random_state=2,
                                                stratify=train_test['low']) # 
            print(train['low'].nunique())
            print(test['low'].nunique())
            test2 = df[df.url.str.contains("dermaamin") == False]
        elif holdout_set == "a12":
            train = df[(df.fitzpatrick==1)|(df.fitzpatrick==2)]
            test = df[(df.fitzpatrick!=1)&(df.fitzpatrick!=2)]
            combo = set(train.label.unique()) & set(test.label.unique())
            print(combo)
            train = train[train.label.isin(combo)].reset_index()
            test = test[test.label.isin(combo)].reset_index()
            train["low"] = train['label'].astype('category').cat.codes
            test["low"] = test['label'].astype('category').cat.codes
        elif holdout_set == "a34":
            train = df[(df.fitzpatrick==3)|(df.fitzpatrick==4)]
            test = df[(df.fitzpatrick!=3)&(df.fitzpatrick!=4)]
            combo = set(train.label.unique()) & set(test.label.unique())
            train = train[train.label.isin(combo)].reset_index()
            test = test[test.label.isin(combo)].reset_index()
            train["low"] = train['label'].astype('category').cat.codes
            test["low"] = test['label'].astype('category').cat.codes
        elif holdout_set == "a56":
            train = df[(df.fitzpatrick==5)|(df.fitzpatrick==6)]
            test = df[(df.fitzpatrick!=5)&(df.fitzpatrick!=6)]
            combo = set(train.label.unique()) & set(test.label.unique())
            train = train[train.label.isin(combo)].reset_index()
            test = test[test.label.isin(combo)].reset_index()
            train["low"] = train['label'].astype('category').cat.codes
            test["low"] = test['label'].astype('category').cat.codes

        level = "high" #9label


        train_path = "temp_train_{}.csv".format(model_name)
        test_path = "temp_test_{}.csv".format(model_name)
        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)

        for indexer, label in enumerate([level]):
            # tensorboard
            writer = SummaryWriter(comment="logs_{}_{}_{}_{}.pth".format(model_name, n_epochs, label, holdout_set))
            print(label)
            weights = np.array(max(train[label].value_counts())/train[label].value_counts().sort_index())
            label_codes = sorted(list(train[label].unique()))
            dataloaders, dataset_sizes = custom_load(
                32,
                10,
                "{}".format(train_path),
                "{}".format(test_path),
                label = label,
                dataset_name = dataset_name)
            print(dataset_sizes)
            # ------------------ 
            # TODO check input size
            model_ft = Network('sparse', [len(label_codes), 6], pretrained=True)
                                                                                                                                                                                 
         
            total_params = sum(p.numel() for p in model_ft.feature_extractor.parameters())
            print('{} total parameters'.format(total_params))
            i = 0
            for p in model_ft.feature_extractor.parameters():
                if(i>=50):
                    p.requires_grad=True
                else:   p.requires_grad=False
                i+=1
            print('i',i)
            total_trainable_params = sum(
                p.numel() for p in model_ft.feature_extractor.parameters() if p.requires_grad)
            print('{} total trainable parameters'.format(total_trainable_params))
            k = 0


            # i = 0
            # for p in bert_model.parameters():
            #     if(i>=(100)):
            #         p.requires_grad=True
            #     else:   p.requires_grad=False
            #     i+=1
            # print('i',i)


            model_ft = model_ft.to(device)

            model_ft = nn.DataParallel(model_ft)
            class_weights = torch.FloatTensor(weights).cuda()
            # criterion = nn.NLLLoss()
            # criterion = nn.CrossEntropyLoss()
            # TODO modify criterion
            criterion = [nn.CrossEntropyLoss(), Confusion_Loss(), 
            nn.CrossEntropyLoss(), Supervised_Contrastive_Loss(0.1, device)]
            optimizer_ft = optim.Adam(list(model_ft.parameters()) , 0.0001) #+list(bert_model.parameters())
            # optimizer_ft = optim.AdamW(model_ft.parameters(),0.0001,weight_decay=0.05)
            exp_lr_scheduler = lr_scheduler.StepLR(
            optimizer_ft,
            step_size=2,
            gamma=0.8)

            print("\nTraining classifier for {}........ \n".format(label))
            print("....... processing ........ \n")
            model_ft, training_results = train_model(
                label,
                dataloaders, device,
                dataset_sizes, model_ft,
                criterion, optimizer_ft,
                exp_lr_scheduler, n_epochs)
            print("Training Complete")
            
            torch.save(model_ft.state_dict(), "model_path_{}_{}_{}_{}.pth".format(model_name, n_epochs, label, holdout_set))
            torch.save(model_ft, "model_path_{}_{}_{}_{}.pt".format(model_name, n_epochs, label, holdout_set))
            print("gold")
            training_results.to_csv("training_{}_{}_{}_{}.csv".format(model_name, n_epochs, label, holdout_set))

            model = model_ft.eval()
            loader = dataloaders["val"]
            prediction_list = []
            fitzpatrick_list = []
            hasher_list = []
            labels_list = []
            p_list = []
            topk_p = []
            topk_n = []
            d1 = []
            d2 = []
            d3 = []
            p1 = []
            p2 = []
            p3 = []
            with torch.no_grad():
                running_corrects = 0
                running_balanced_acc_sum  = 0
                total = 0
                for i, batch in enumerate(dataloaders['val']):
                    inputs = batch["image"].to(device)
                    classes = batch[label].to(device)
                    fitzpatrick = batch["fitzpatrick"]  # skin type
                    hasher = batch["hasher"]
                    outputs = model(inputs.float())  # (batchsize, classes num)
                    probability = torch.nn.functional.softmax(outputs[0], dim=1)
                    ppp, preds = torch.topk(probability, 1) #topk values, topk indices
                    if label == "low":
                        _, preds5 = torch.topk(probability, 3)  # topk values, topk indices
                        # topk_p.append(np.exp(_.cpu()).tolist())
                        topk_p.append((_.cpu()).tolist())
                        topk_n.append(preds5.cpu().tolist())
                    running_balanced_acc_sum += balanced_accuracy_score(classes.data.cpu(), preds.reshape(-1).cpu()) * inputs.shape[0]
                    running_corrects += torch.sum(preds.reshape(-1) == classes.data)
                    p_list.append(ppp.cpu().tolist())
                    prediction_list.append(preds.cpu().tolist())
                    labels_list.append(classes.tolist())
                    fitzpatrick_list.append(fitzpatrick.tolist())
                    hasher_list.append(hasher)
                    total += inputs.shape[0]
                acc = float(running_corrects)/float(dataset_sizes['val'])
                balanced_acc = float(running_balanced_acc_sum)/float(dataset_sizes['val'])
            if label == "low":
                for j in topk_n: # each sample
                    for i in j:  # in k
                        d1.append(i[0])
                        d2.append(i[1])
                        d3.append(i[2])
                for j in topk_p:
                    for i in j:
                        # print(i)
                        p1.append(i[0])
                        p2.append(i[1])
                        p3.append(i[2])
                df_x=pd.DataFrame({
                                    "hasher": flatten(hasher_list),
                                    "label": flatten(labels_list),
                                    "fitzpatrick": flatten(fitzpatrick_list),
                                    "prediction_probability": flatten(p_list),
                                    "prediction": flatten(prediction_list),
                                    "d1": d1,
                                    "d2": d2,
                                    "d3": d3,
                                    "p1": p1,
                                    "p2": p2,
                                    "p3": p3})
            else:
                # print(len(flatten(hasher_list)))
                # print(len(flatten(labels_list)))
                # print(len(flatten(fitzpatrick_list)))
                # print(len(flatten(p_list)))
                # print(len(flatten(prediction_list)))
                df_x=pd.DataFrame({
                                    "hasher": flatten(hasher_list),
                                    "label": flatten(labels_list),
                                    "fitzpatrick": flatten(fitzpatrick_list),
                                    "prediction_probability": flatten(p_list),
                                    "prediction": flatten(prediction_list)})
            df_x.to_csv("results_{}_{}_{}_{}.csv".format(model_name, n_epochs, label, holdout_set),
                            index=False)
            print("\n Accuracy: {}  Balanced Accuracy: {} \n".format(acc, balanced_acc))
        print("done")
        # writer.close()
