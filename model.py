import torch
import pytorch_lightning as pl
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support
from resnet import ResNet18 

# TODO delete imports after cross val 
#import hickle 
#from sklearn.model_selection import train_test_split
#from team_code import collate_fn
#from pytorch_lightning.loggers import WandbLogger
#import wandb

class PCGClassifier(pl.LightningModule):
    def __init__(self, mu=1.0, epsilon=1.0):
        super().__init__()
        #CHANGE FOR SINGLE  
        self.img_encoder =ResNet18(4) #nn.Sequential(nn.Conv2d(5, 1, 1, stride=4), nn.Flatten(), nn.Linear(3136, 3))#ResNet18(5) #TODO define this elsewhere
        self.img_encoder.linear = nn.Identity() 
        self.murmur_clf = nn.Linear(12544, 3) #MAGIC NUMBER change the 49
        self.outcome_clf = nn.Linear(12544,2)
        #self.clf_layer = nn.Linear(512, 3) #E.g. if hidden dimension is 512, go to 3: present, not present, unsure
        self.loss_fn = nn.CrossEntropyLoss()
        self.batch_size=10
        self.sftmax = nn.Softmax(dim=1)
        self.mu=mu
        self.epsilon=epsilon

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def forward(self, x):
        #this is independent of training step
        #basically use this for prediction
        x = self.img_encoder(x)
        
        # Murnmur classifier  
        xmurm = self.murmur_clf(x) 
        
        # Outcome Classifier 
        xoutcome=self.outcome_clf(x)
        #print("X SHAPE", x.shape)
        #x = self.clf_layer(x)
        return xmurm, xoutcome

    def training_step(self, batch, batch_idx):
        x, ymurm, yout = batch #This is data and label
        #maybe some resizing happens here
        xmurm, xoutcome = self(x) #same as calling my_model(...) (i.e. calls forward)
        murmloss = self.loss_fn(xmurm, ymurm)
        outcomeloss=self.loss_fn(xoutcome, yout)
        loss = (self.mu*murmloss)+(self.epsilon*outcomeloss)
        
        murmpreds = xmurm#.argmax(dim=1)
        outpreds  = xoutcome#.argmax(dim=1)
        #print("OUTPERDS", outpreds, '------', outpreds.argmax(dim=1))
        self.log("train_loss", loss, on_step=False, on_epoch=True, logger=False)
        return {'loss':loss, 'outcome_preds':outpreds.detach().cpu().numpy(),'murmur_preds':murmpreds.detach().cpu().numpy(), 'outcome_labels':yout.detach().cpu().numpy(), 'murmur_labels':ymurm.detach().cpu().numpy(), 'murmur_probs':self.sftmax(murmpreds).detach().cpu().numpy(), 'outcome_probs':self.sftmax(outpreds).detach().cpu().numpy()}

    def test_step(self, batch, batch_idx):
        x, ymurm, yout = batch #This is data and label
        #maybe some resizing happens here
        xmurm, xoutcome = self(x) #same as calling my_model(...) (i.e. calls forward)
        murmloss = self.loss_fn(xmurm, ymurm)
        outcomeloss=self.loss_fn(xoutcome, yout)
        loss = (self.mu*murmloss)+(self.epsilon*outcomeloss)
        #print("XMURM SHAPE", xmurm.shape, "OUTCOME SHAPE", xoutcome.shape)
        murmpreds = xmurm#.argmax(dim=1)
        outpreds  = xoutcome#.argmax(dim=1)
        self.log("val_loss", loss, on_step=False, on_epoch=True, logger=False)
        #self.log('outcome_prf', precision_recall_fscore_support(outpreds.argmax(dim=1).detach().cpu().numpy(), yout.detach().cpu().numpy()), on_step=False, on_epoch=True, logger=False)
        #self.log('murm_prf', precision_recall_fscore_support(murmpreds.argmax(dim=1).detach().cpu().numpy(), ymurm.detach().cpu().numpy()), on_step=False, on_epoch=True, logger=False)
        return {'loss':loss, 'outcome_preds':outpreds.detach().cpu().numpy(),'murmur_preds':murmpreds.detach().cpu().numpy(), 'outcome_labels':yout.detach().cpu().numpy(), 'murmur_labels':ymurm.detach().cpu().numpy(), 'murmur_probs':self.sftmax(murmpreds).detach().cpu().numpy(), 'outcome_probs':self.sftmax(outpreds).detach().cpu().numpy()}

    def test_epoch_end(self, outputs):
        # do something with the outputs of all test batches
        print(outputs)
        outcome_preds = outputs['outcome_preds']
    
        #some_result = calc_all_results(all_test_preds)
        #self.log(some_result) 

    def predict_step(self, batch, batch_idx):
        x = batch
        #print('x', x.shape)
        xmurm, xoutcome = self(x)
        #loss = self.loss_fn(x_hat, y)
        #print(x_hat.shape)
        murmpreds = xmurm#.argmax(dim=1)
        outpreds  = xoutcome#.argmax(dim=1)
        #print("OUTPREDAS", outpreds)
        #self.log("pred_loss", loss, on_epoch=True, on_step=False, logger=False)
        return {'outcome_preds':outpreds.detach().cpu().numpy(),'murmur_preds':murmpreds.detach().cpu().numpy(), 'murmur_probs':self.sftmax(murmpreds).detach().cpu().numpy(), 'outcome_probs':self.sftmax(outpreds).detach().cpu().numpy()}

    def log_metrics(self, outputs, split):
        #print('losses', [l['loss'] for l in outputs])
        loss = sum(l['loss'].item() for l in outputs)
        preds = torch.cat([o['preds'] for o in outputs]).detach().cpu()
        labels = torch.cat([o['labels'] for o in outputs]).detach().cpu()
        #print(loss, preds.shape, labels.shape)
        #print(preds, labels)
        acc = torch.sum(preds==labels)/len(labels)
        precision, recall, f1, support =  precision_recall_fscore_support(labels, preds, average = 'macro', zero_division=0)
        #print('LOSS', loss, 'p', precision, 'r', recall, 'f1', f1, 'acc', acc)
        self.log('', {f'{split}_accuracy':acc, f'{split}_f1':f1, f'{split}_recall':recall, f'{split}_precision':precision}, logger=False) 

    def training_epoch_end(self, outputs):
        pass#self.log_metrics(outputs, split='train')
    def validation_epoch_end(self, outputs):
        pass#self.log_metrics(outputs, split='validation')
    def test_epoch_end(self, outputs):
        pass#self.log_metrics(outputs, split='test')

class Dataloader(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx][0], torch.tensor(self.data[idx][1]).long()


def collate_fn(batch): 
    patient_imgs = [f["patient_imgs"] for f in batch]
    labels = [f["label"] for f in batch]
    patient_imgs = torch.stack(patient_imgs)
    labels = torch.stack(labels).argmax(dim=1)
    outputs = {"patient_imgs": patient_imgs, "labels": labels}
    return tuple(outputs.values())
    
    #if __name__ == "__main__":
    
    #wandb.init(project='cinc2022', name='test')
    # Load pre-processed data from hickle 
    #data = hickle.load("preprocessed_data.hickle")
    # Random train & val split 
    #train, val = train_test_split(data, test_size=0.3)
    #train_loader = DataLoader(train, shuffle=True, batch_size=64, collate_fn=collate_fn) 
    #val_loader = DataLoader(val, shuffle=True, batch_size=64, collate_fn=collate_fn)
     
    #model = PCGClassifier()
    #training_data = DataLoader(Dataloader([(torch.rand(10), torch.randint(0,2, (1,))[0]) for i in range(11)]), batch_size=11, shuffle=False)#None #TODO get this in a dataloader 
    #wandb_logger = WandbLogger(project="cinc2022")
    # TODO how do use a metric here?  
    #trainer=pl.Trainer(gpus=1, logger = wandb_logger, max_epochs=64)
    #trainer.validate(model, training_data)
    #trainer.fit(model, train_loader, val_loader)
      
    
