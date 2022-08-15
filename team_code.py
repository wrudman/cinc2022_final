#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Import libraries and functions. You can change or remove them.
#
################################################################################
#import matplotlib.pyplot as plt
from helper_code import *
import torch
from torch.utils.data import DataLoader
import numpy as np, scipy as sp, scipy.stats, os, sys
from make_graphs import Cinc_Graphs
import pytorch_lightning as pl
#TODO change to single channel where required
from model import PCGClassifier
from pytorch_lightning.callbacks import ModelCheckpoint

from helper_code import *
import numpy as np, scipy as sp, scipy.stats, os, sys, joblib

print("importing pil")
# TODO ADD TO REQUIREMENTS OR REMOVE
from PIL import Image

#DELETE IMPORTS AFTER USE
#from rich.traceback import install
#install()
#import os
#import glob
#TODO remove
#import hickle
#from random import shuffle 
from model_single import PCGClassifier_Single
#import tracemalloc
#import wandb
#from pytorch_lightning.loggers import WandbLogger
#import matplotlib.pyplot as plt

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments.
#
################################################################################

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    # Find data files.
    #if verbose >= 1:
    #print('Finding data files...')
    #tracemalloc.start()
    
    # Find the patient data files.
    patient_files = find_patient_files(data_folder)
    num_patient_files = len(patient_files)
   
    if num_patient_files==0:
        raise Exception('No data was provided.')

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    murmur_classes = ['Present', 'Unknown', 'Absent']
    num_murmur_classes = len(murmur_classes) 
    outcome_classes = ['Abnormal', 'Normal'] 
    num_outcome_classes = len(outcome_classes) 
    location_order = ['AV', 'PV', 'TV', 'MV','Phc']
    #location_order = ['AV', 'PV', 'TV', 'MV'] 
    
     
    data_dict = {}
    # Extract the features and labels.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...') 

    #for i in range(10):
     
    for i in range(num_patient_files):      
        data_dict[i] = {} 
        if verbose >= 2:
            print('    {}/{}...'.format(i+1, num_patient_files))

        # Load the current patient data and recordings.
        current_patient_data = load_patient_data(patient_files[i]) 
        current_recordings = load_recordings(data_folder, current_patient_data)  
        #print("Num recordings", len(current_recordings)) 
        # Create images
        #TODO MAKE SURE THE CORRECT NUMBER OF CHANNELS ARE BEING MADE    
        patient_imgs = make_img_single(current_patient_data, current_recordings)  
         
        # Extract labels and use one-hot encoding.
        current_murmur_labels = np.zeros(num_murmur_classes, dtype=int)
        current_outcome_labels = np.zeros(num_outcome_classes, dtype=int)
        
        # THIS GETS LABELS FOR EACJ TASK
        murmur_label = get_murmur(current_patient_data) 
        outcome_label = get_outcome(current_patient_data)
        
        if murmur_label in murmur_classes:  
            j = murmur_classes.index(murmur_label)
            current_murmur_labels[j] = 1   
        
        if outcome_label in outcome_classes:
            j = outcome_classes.index(outcome_label)
            current_outcome_labels[j] = 1 
        
        data_dict[i]["patient_imgs"] = patient_imgs.float() 
        data_dict[i]["murmur_label"] = torch.tensor(current_murmur_labels, dtype=torch.long) 
        data_dict[i]["outcome_label"] = torch.tensor(current_outcome_labels, dtype=torch.long) 
    
    #print("Tensor size",data_dict[i]['patient_imgs'].shape) 
    
    # MEMORY Debugging 
    #current, peak = tracemalloc.get_traced_memory()
     
    #print("Preprocessing Stats")
    #print("Current Mem in GB:", current*1e-9)
    #print("Preak Mem in GB:", peak*1e-9)
    
    #tracemalloc.stop()

    # Comment out for submission
   
     
    #data_dict = list(hickle.load('pre_train_imgs.hickle').values()) 
   
    print("...Begin Training") 
    
    #tracemalloc.start()
    # TODO: Change after pre-train 
    train_loader = DataLoader(data_dict, batch_size=20, shuffle=False, collate_fn = collate_fn) 
    
    #wandb_logger = WandbLogger(project="cinc2022", name='model1.0') 
    # I changed it to model_folder bc that's the dir they'll make 
    trainer = pl.Trainer(gpus=1, max_epochs=5, callbacks=[ModelCheckpoint(dirpath=model_folder, filename='best', monitor='train_loss', mode='min')])
   
    # LOAD MODEL AFTER PRE-TRAINED 
    #TODO: Make sure we 
    
    #net = load_pretrained_model(model_folder, verbose)
    #net = load_challenge_model(model_folder, verbose)
    #TODO: Make sure we have the correct classifier here.  
    net = PCGClassifier_Single()  
    #net = PCGClassifier()
    net.load_from_checkpoint("best_single.ckpt") 
    net.train()
    trainer.fit(net, train_loader)
     
    return None

# Load your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.


def load_pretrained_model(model_folder, verbose):
    if os.path.exists(os.path.join(model_folder, "best_mini.ckpt")):
        filename = os.path.join(model_folder, 'best_mini.ckpt')
    else:
        filename = os.path.join(model_folder, 'best_mini-v1.ckpt')
    model = PCGClassifier.load_from_checkpoint(filename)
    return model#joblib.load(filename)

def load_challenge_model(model_folder, verbose):
    if os.path.exists(os.path.join(model_folder, "best-v1.ckpt")):
        filename = os.path.join(model_folder, 'best-v1.ckpt')
    else:
        filename = os.path.join(model_folder, 'best.ckpt')
    model = PCGClassifier.load_from_checkpoint(filename)
    return model#joblib.load(filename)

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_model(model, data, recordings, verbose):
     
    sample = make_img(data, recordings)  
    test_dl = DataLoader([sample], batch_size=1) 
    murmur_classes = ['Present', 'Unknown', 'Absent']
    outcome_classes= ['Abnormal', 'Normal']
    trainer = pl.Trainer(gpus=1)
    preds = trainer.predict(model, test_dl)[0]
    print("Preds", preds)
    murmur_probabilities = preds['murmur_probs'][0]
    outcome_probabilities= preds['outcome_probs'][0]
    if verbose:
        print('probabilities', murmur_probabilities, 'classes', murmur_classes, outcome_probabilities, outcome_classes)
    
    # Choose label with highest probability.
    def assign_label(probabilities, classes):
        labels = np.zeros(len(classes), dtype=np.int_)
        idx = np.argmax(probabilities)
        labels[idx] = 1 #(N,3), 
        return labels
    murmur_labels = assign_label(murmur_probabilities, murmur_classes)
    outcome_labels= assign_label(outcome_probabilities, outcome_classes)
    classes = murmur_classes+outcome_classes
    labels = np.concatenate((murmur_labels, outcome_labels))
    probabilities = np.concatenate((murmur_probabilities, outcome_probabilities))
    #print("Classes", classes)
    #print("Labels", labels)
    #print("Probabilities", probabilities)
    return classes, labels, probabilities #['pos', 'unknown',...], (N,3), (N,3)

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################
#  JACK AND WILLIAM ORIGINAL FUNCTIONS 
def collate_fn(batch):
    patient_imgs = [f["patient_imgs"] for f in batch]
    murmur_labels = [f['murmur_label'] for f in batch] 
    outcome_labels = [f['outcome_label'] for f in batch]
    
    patient_imgs = torch.stack(patient_imgs)
    murmur_labels = torch.stack(murmur_labels).argmax(dim=1)
    outcome_labels = torch.stack(outcome_labels).argmax(dim=1)

    #print('labelsssss', labels)
    outputs = {"patient_imgs": patient_imgs, "murmur_labels": murmur_labels, "outcome_labels": outcome_labels}
    return tuple(outputs.values())


def collate_pretrain(batch):
    patient_imgs = [f["patient_imgs"].view(1,224,224).repeat(4,1,1) for f in batch]
    outcome_labels = [f['label'] for f in batch] 
    murmur_labels = [torch.tensor([1,0,0], dtype=torch.long) if torch.equal(f['label'], torch.tensor([1,0])) == True else torch.tensor([0,0,1], dtype=torch.long) for f in batch]
    
    patient_imgs = torch.stack(patient_imgs)
    murmur_labels = torch.stack(murmur_labels).argmax(dim=1)
    outcome_labels = torch.stack(outcome_labels).argmax(dim=1)

    #print('labelsssss', labels)
    outputs = {"patient_imgs": patient_imgs, "murmur_labels": murmur_labels, "outcome_labels": outcome_labels}  
    return tuple(outputs.values())


def collate_pretrain_single(batch):
    patient_imgs = [f["patient_imgs"].view(1,224,224) for f in batch]
    outcome_labels = [f['label'] for f in batch] 
    murmur_labels = [torch.tensor([1,0,0], dtype=torch.long) if torch.equal(f['label'], torch.tensor([1,0])) == True else torch.tensor([0,0,1], dtype=torch.long) for f in batch]
    
    patient_imgs = torch.stack(patient_imgs)
    murmur_labels = torch.stack(murmur_labels).argmax(dim=1)
    outcome_labels = torch.stack(outcome_labels).argmax(dim=1)

    #print('labelsssss', labels)
    outputs = {"patient_imgs": patient_imgs, "murmur_labels": murmur_labels, "outcome_labels": outcome_labels}  
    return tuple(outputs.values())

def make_img(data, recordings): 
     
     location_order = ['AV', 'PV', 'TV', 'MV', 'Phc']     
     #Get the indices of current locations to use for padding 
     locations = get_locations(data)
        
     location_indx = [location_order.index(x) for x in locations]  
        
     # Empty tensor that will be updated with img locations 
     patient_imgs = torch.zeros(4, 224, 224)
     #print(locations)
     #print(location_indx) 
     # Get frequency of the recording to pre-process signal
     fs = get_frequency(data)   
     # here we go through all recordings for a patient, make an image per recording and update patient_imgs with recordings.  
     for j in range(len(recordings)):  
         if location_indx[j] == 4: 
             continue 
         rec = recordings[j]
         g = Cinc_Graphs()
         img = g.make_tde(rec, fs) 
         patient_imgs[location_indx[j]] = img
     #out_img = np.transpose(np.array(patient_imgs)*255, (1,2,0)).astype(np.uint8)
     #myimg = Image.fromarray(out_img, mode='RGBA')
     #label = get_murmur(data)
     #myimg.save(f"img_{label}_" + indx + "_.png")
     return patient_imgs

def collate_test(batch):
    patient_imgs = batch 
    outputs = {"patient_imgs": patient_imgs, "labels": torch.zeros(4).long()}
    return tuple(outputs.values())

# OLD CHALLENGE FUNCTIONS 
# Save your trained model.
#def save_challenge_model(model_folder, classes, imputer, classifier):
#    d = {'classes': classes, 'imputer': imputer, 'classifier': classifier}
#    filename = os.path.join(model_folder, 'model.sav')
#    joblib.dump(d, filename, protocol=0)

# Extract features from the data.
def get_features(data, recordings):
    # Extract the age group and replace with the (approximate) number of months for the middle of the age group.
    age_group = get_age(data)

    if compare_strings(age_group, 'Neonate'):
        age = 0.5
    elif compare_strings(age_group, 'Infant'):
        age = 6
    elif compare_strings(age_group, 'Child'):
        age = 6 * 12
    elif compare_strings(age_group, 'Adolescent'):
        age = 15 * 12
    elif compare_strings(age_group, 'Young Adult'):
        age = 20 * 12
    else:
        age = float('nan')

    # Extract sex. Use one-hot encoding.
    sex = get_sex(data)

    sex_features = np.zeros(2, dtype=int)
    if compare_strings(sex, 'Female'):
        sex_features[0] = 1
    elif compare_strings(sex, 'Male'):
        sex_features[1] = 1

    # Extract height and weight.
    height = get_height(data)
    weight = get_weight(data)

    # Extract pregnancy status.
    is_pregnant = get_pregnancy_status(data)

    # Extract recording locations and data. Identify when a location is present, and compute the mean, variance, and skewness of
    # each recording. If there are multiple recordings for one location, then extract features from the last recording.
    locations = get_locations(data)

    recording_locations = ['AV', 'MV', 'PV', 'TV', 'PhC']
    num_recording_locations = len(recording_locations)
    recording_features = np.zeros((num_recording_locations, 4), dtype=float)
    num_locations = len(locations)
    num_recordings = len(recordings)
    if num_locations==num_recordings:
        for i in range(num_locations):
            for j in range(num_recording_locations):
                if compare_strings(locations[i], recording_locations[j]) and np.size(recordings[i])>0:
                    recording_features[j, 0] = 1
                    recording_features[j, 1] = np.mean(recordings[i])
                    recording_features[j, 2] = np.var(recordings[i])
                    recording_features[j, 3] = sp.stats.skew(recordings[i])
    recording_features = recording_features.flatten()
    features = np.hstack(([age], sex_features, [height], [weight], [is_pregnant], recording_features))

    return np.asarray(features, dtype=np.float32)  

def make_img_single(data, recordings): 
     location_order = ['AV', 'PV', 'TV', 'MV','Phc']     
     #Get the indices of current locations to use for padding 
     locations = get_locations(data)
         
     # Get frequency of the recording to pre-process signal
     fs = get_frequency(data)   
     # here we go through all recordings for a patient, make an image per recording and update patient_imgs with recordings.  
     #print(recordings) 
     if len(recordings) > 1: 
        rec = np.concatenate(np.array(recordings)) 
     else:
         rec = np.array(recordings[0]) 
     #print(len(recordings)) 
     g = Cinc_Graphs() 
     patient_imgs = g.make_tde(rec, fs) 
     #print("SHAPE: ", patient_imgs.shape) 
     return patient_imgs

def main(): 
    """ TRAIN """  
    #TODO: EITHER REMOVE OR ADD TO REQUIREMENTS 
    #from PIL import Image
    #import numpy as np
    #print("can we import cv2?") 
    #import cv2
    #print("YES! We can!") 
    data_folder = '../2cinc2022/data/physionet.org/files/circor-heart-sound/1.0.3/training_data'   
    model_folder = 'models'
    print("Trying to train") 
    train_challenge_model(data_folder, model_folder, verbose=1)
    #val_data = hickle.load("val_butter_data.hickle")
     
    #pre_data_folder = 'physionet.org/files/challenge-2016/1.0.0/training' 
    
    def make_dict(pre_data_folder): 
        data_dict = {} 
        os.chdir(pre_data_folder) 
        g = Cinc_Graphs() 
        counter = 0 
        
        for patient_files in glob.glob("*.hea"):            
        # Load the current patient data and recordings.
            current_patient_data = load_patient_data(patient_files)  
            cp_wav = current_patient_data.split(" ")[0] + '.wav' 
            current_recording, fs = load_wav_file(cp_wav) 
            # Create images
            patient_img =  g.make_tde(current_recording, fs)    
            # THIS GETS LABELS FOR EACJ TASK
            murmur_label = get_pre_train_label(current_patient_data)  
            data_dict[counter] = {}
            data_dict[counter]["patient_imgs"] = patient_img 
            data_dict[counter]["label"] = murmur_label 
            counter += 1
        
        return data_dict 

    #pre_data_folder = 'physionet.org/files/challenge-2016/1.0.0/training' 
    #data_dict = make_dict(pre_data_folder)    
    #os.chdir("/gpfs/data/ceichoff/cinc2022_final")
    #hickle.dump(data_dict, "pre_train_imgs.hickle", mode='w')  
    #wandb.init('cinc2022',name='model1')
    

if __name__ == '__main__':
    main()
    model_dir = sys.argv[1]
    data_dir = sys.argv[2]
    output_dir = sys.argv[3]
    run_challenge_model(sys.argv[1], sys.argv[2], sys.argv[3], True)

