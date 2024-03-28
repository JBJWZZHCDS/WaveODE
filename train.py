import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.utils as utils
import torch.nn.functional as F
import torchaudio
import os
from tqdm import tqdm,trange
import matplotlib.pyplot as plt
import numpy as np  

from params import params
from dataset import AudioMelSet
from models import Velocity
from params import params

    
def train(mixTraining=params['trainWithHybridPrecision']):
    
    trainData=AudioMelSet(params['trainAudiosPath'],params['trainMelsPath'])
    trainLoader=torch.utils.data.DataLoader(trainData,batch_size=params['trainBatch'],shuffle=True,pin_memory=True,num_workers=8)

    device=params['trainDevice']
    gamma=params['trainGamma']
    betas=params['trainBetas']
    weightDecay=params['trainWeightDecay']

    velocity=Velocity().to(device)

    vOptimizer=optim.AdamW(velocity.parameters(),lr=params['trainLearnRateVelocity'],
                           betas=betas,weight_decay=weightDecay)
    
    if os.path.exists(params['trainCheckPointPath']):
        
        all=torch.load(params['trainCheckPointPath'])

        velocity.load_state_dict(all['velocity'],strict=False)

        vOptimizer.load_state_dict(all['vOptimizer'])     
        
        nowStep=all['step']
        nowEpoch=all['epoch']
        
        for param_group in vOptimizer.param_groups:
            param_group['weight_decay']=weightDecay
        for param_group in vOptimizer.param_groups:
            param_group['betas']=betas

    else:
        
        nowStep=0
        nowEpoch=0
        
        path=params['trainCheckPointPath']
        for para in velocity.parameters():
            para.data.clamp_(-0.25,0.25)
            
        pos=path.rfind('_')
        if pos==-1 or pos==len(path)-1 or path[pos+1:].isdigit()==False:
            path=path+'_'+str(nowStep)
        else:
            path=path[:pos]+'_'+str(nowStep)
        
        torch.save({
                    'velocity':velocity.state_dict(),
                    'vOptimizer':vOptimizer.state_dict(),
                    'step':nowStep,
                    'epoch':nowEpoch,
                    },path)
        


    scaler=torch.cuda.amp.GradScaler(enabled=mixTraining)
    melProcessor = torchaudio.transforms.MelSpectrogram(
                        sample_rate=params['sampleRate'],
                        n_fft=params['fftSize'],
                        win_length=params['windowSize'],
                        hop_length=params['hopSize'],
                        n_mels=params['melBands'],
                        f_min=params['fmin'],
                        f_max=params['fmax']
                    ).to(device)
    
    maximunEnergy=torch.sqrt(torch.tensor(params['melBands']*32768.0))
    meanMelLoss=None
    meanVelocityLoss=None
    meanVelocityL1Loss=None
    
    while True:
        
        tqdmLoader=tqdm(trainLoader,desc=f'train Epoch: {nowEpoch}, starting step={nowStep}')
        for (audios,mels) in tqdmLoader:
                
            with torch.cuda.amp.autocast(enabled=mixTraining):
                
                x1=audios.to(device)
                mels=mels.to(device)
                
                energy=(2**mels).sum(dim=1).sqrt().unsqueeze(1)
                sigma=(energy/maximunEnergy).repeat_interleave(repeats=params['hopSize'],dim=2)

                epsilon=torch.randn_like(sigma)
                x0=sigma*epsilon

                t=torch.rand(x0.size(0),1,1).to(device)

                xt=x0*(1-t)+x1*t
                dx_dt=x1
                timeScale=1.0/(1.0-t.clamp(max=0.9))
               
                predict=velocity(xt,mels,t)
                fakeMels=(melProcessor(predict)+1).log2()
                realMels=(melProcessor(x1)+1).log2()
                melLoss=(fakeMels-realMels).abs().mean()
                
                delta=(predict-dx_dt)
                velocityL1Loss=delta.abs().mean()
                velocityLoss=(timeScale*delta.pow(2)).mean()
                loss=velocityLoss+0.005*melLoss+0.02*velocityL1Loss
                
                if meanMelLoss==None:
                    meanMelLoss=melLoss.item()
                    meanVelocityLoss=velocityLoss.sqrt().item()
                    meanVelocityL1Loss=velocityL1Loss.item()
                else:
                    meanMelLoss=meanMelLoss*0.99+0.01*melLoss.item()
                    meanVelocityLoss=meanVelocityLoss*0.99+0.01*velocityLoss.sqrt().item()
                    meanVelocityL1Loss=meanVelocityL1Loss*0.99+0.01*velocityL1Loss.item()
                
                tqdmLoader.set_postfix(L1Loss=round(meanVelocityL1Loss,4),
                                       ODELoss=round(meanVelocityLoss,4),
                                       MelLoss=round(meanMelLoss,4))
                
                vOptimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(vOptimizer)
                scaler.step(vOptimizer)
                scaler.update()                

                nowStep+=1
            
                if nowStep%params['trainCheckPointSavingStep']==0:
                
                    path=params['trainCheckPointPath']
                    pos=path.rfind('_')
                    if pos==-1 or pos==len(path)-1 or path[pos+1:].isdigit()==False:
                        path=path+'_'+str(nowStep)
                    else:
                        path=path[:pos]+'_'+str(nowStep)

                    torch.save({
                                'velocity':velocity.state_dict(),
                                'vOptimizer':vOptimizer.state_dict(),
                                'step':nowStep,
                                'epoch':nowEpoch,
                                },path)
       

                if nowStep%params['trainLearnRateDecayStep']==0:
                    for param_group in vOptimizer.param_groups:
                        param_group['lr']*=gamma
                       
                if nowStep>=params['trainSteps']:
                    return    
                       
        nowEpoch+=1
            

if __name__=='__main__':
    train()
    
