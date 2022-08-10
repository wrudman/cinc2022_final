import pickle
import glob
import os
import wfdb
from wfdb import processing
import numpy as np
import heartpy as hp 
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA 
import scipy.io as sio
import scipy.signal as sig
from sklearn import manifold 
import re
from helper_code import *
import torch
from torchvision.utils import save_image
from PIL import Image
from torchvision import transforms
def preprocess(window, fs, is_spec=False):			
    # Remove baseline_wander and resample signal to target frequency  
    fs_target = 1000			
    pcg = np.trim_zeros(window) 
    #print("trim_zero", pcg)   
    pcg, _ = processing.resample_sig(pcg,fs=fs,fs_target=fs_target)		
    #print("resample sig", pcg) 
    pcg = wfdb.processing.normalize_bound(pcg)   
    butter = sig.butter(20, 100, btype='lowpass', fs=1000, output='sos') 
    pcg = sig.sosfilt(butter,pcg) 
    #print(pcg)
    #print(pcg.tolist())
    return pcg.tolist()


class Cinc_Graphs: 
    """Class containing all the functions to make the the four types of graphs considred in our project: time delay embedding graphs, spectrograms,
    processed signal graphs and LSTMEmbedding graphs. """	
    def __init__(self):	
        self.window_length = 60 
        self.slide_length = 1
        self.pc = 2
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.0], std=[1.0])
        ])
    def time_delay_embedding(self, data):	
        """Creates a time delay embedding from the data contained in a SINGLE window to create a representation of the quasi-attractor. Next, principal 
        component analysis is performed on quasi-attractor in order to reduce the number of graphs we need to produce.
        INPUTS:
        data: the ecg signal data of a single window. 
        window_length: determines the size of the embedding
        slide_length: factor by which we move our window
        pc: specifies the number of principal components of the time delay embedding that we want to consider.	
        RETURNS: 5 principal components of each time delay vector generated in data (10800 x 5)"""	
        def sliding_window(self, data):
            """HELPER funciton to create the windows of window_length by moving the window in slide_length steps."""	
            windows=[]
            d=0
            for i in range(len(data)):
                if len(data[d:self.window_length+d]) < self.window_length:
                    break
                else:
                    windows.append(data[d:self.window_length+d])
                d+=self.slide_length
            return windows	
        tde = np.array(sliding_window(self,data))	
        return tde

    def make_tde(self, lead, fs):  
        """Plots the first and second principal components of the time-delay embedding
        of the signal each window of windows and saves it in target_director
        INPUTS: windows: ECG Signals 
        RETURNS: None"""	
        plt.rcParams['lines.linewidth'] = 0.05 
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color='k')	
        px = 1/plt.rcParams['figure.dpi']	
        #Preprocess windows, create time delay embedding and perform the specified method of dimens			
        if len(lead) == 0: return None 		
        tde = self.time_delay_embedding(preprocess(lead,fs)) 	
        if len(tde) == 0: return None	
        tde_coords= PCA(n_components=2).fit_transform(tde)	 
        fig = plt.figure(figsize=(224*px,224*px))
        plt.plot(tde_coords[:,0],tde_coords[:,1])
        fig.canvas.draw()
        #img_array = np.fromstring(fig.canvas.tostring_rgb(),dtype=np.uint8,sep='')					
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        #print(shape.shape)
        plt.close()  
        #print(np.eshape(img_array, (3,225,225)))
        #shape = np.reshape(img_array, (3,225,225))
        #print(np.array_equal(shape[0], shape[1]))        
        #from PIL import Image
        #img = Image.fromarray(shape[:,:,0], mode='L')
        #img.save("0.png")
        #img = Image.fromarray(shape[:,:,1], mode='L')
        #img.save("1.png")
        #img = Image.fromarray(shape[:,:,2], mode='L')
        #img.save("2.png")
        #plt.savefig(fname)	
        #plt.close()	
        return self.transform(img_array[:,:,0])
    
    def make_tde2(self, lead, fs):  	 
        #Preprocess windows, create time delay embedding and perform the specified method of dimens			
        HEIGHT= 224 #480
        WIDTH=224 #640
        marker_weight = 51*2 #how 'dark' to make a point in the image
        def create_img_axis(data, size):
            dmax = data.max()
            dmin = data.min()
            step = (dmax-dmin)/size
            #print(dmin,dmax, step)
            bins = np.arange(dmin, dmax, step)[:size]
            #print(size, len(bins),bins[0], bins[-1])
            idxs = np.digitize(data, bins)-1 #minus 1 because digitize starts at 1...
            return idxs#axis	
         
        tde = self.time_delay_embedding(preprocess(lead,fs)) 	 
        #if len(tde) == 0: return None	
        tde_coords= PCA(n_components=2).fit_transform(tde)		
        x_axis = create_img_axis(tde_coords[:,0], WIDTH)
        y_axis = create_img_axis(tde_coords[:,1], HEIGHT)
        img = np.ones((WIDTH, HEIGHT))*255
        for x, y in zip(x_axis, y_axis):
            img[x,y]=max(0, img[x,y]-marker_weight) #clamp to 0
            img[max(0,x-1),y]=max(0, img[max(0,x-1),y]-2*16)
            img[max(0,x-1),max(0,y-1)]=max(0, img[max(0,x-1),max(0,y-1)]-2*16)	
            img[min(WIDTH-1,x+1),y]=max(0, img[min(WIDTH-1,x+1),y]-2*16)
            img[min(WIDTH-1,x+1),min(HEIGHT-1,y+1)]=max(0, img[min(WIDTH-1,x+1),min(HEIGHT-1,y+1)]-2*16)
            img[max(0,x-1),max(0,y-1)]=max(0, img[max(0,x-1),max(0,y-1)]-2*16)
            img[max(WIDTH-1,x-1),y] = max(0,img[max(WIDTH-1, x-1),y-2*16]) 
            img[x,max(HEIGHT-1,y-1)] = max(0,img[x,max(HEIGHT-1,y)-2*16])
            img[max(WIDTH-1,x-1),max(HEIGHT-1,y-1)] = max(0,img[max(WIDTH-1,x-1),max(HEIGHT-1,y)-2*16])	
        #TODO: make images between 0-1	 
        return torch.tensor(img)
		
