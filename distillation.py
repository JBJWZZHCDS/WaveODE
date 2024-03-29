import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.utils as utils
import torch.nn.functional as func
import torchaudio
import os
from tqdm import tqdm,trange
import matplotlib.pyplot as plt
import numpy as np  

from params import params
from dataset import AudioMelSet
from models import Velocity
from params import params
from torchdiffeq import odeint
    
def distillation(mixTraining=params['distillWithHybridPrecision']):
    
    trainData=AudioMelSet(params['distillAudiosPath'],params['distillMelsPath'])
    trainLoader=torch.utils.data.DataLoader(trainData,
                                            batch_size=params['distillBatch'],
                                            shuffle=True,pin_memory=True,
                                            num_workers=8)

    device=params['distillDevice']
    betas=params['distillBetas']
    weightDecay=params['distillWeightDecay']
    gamma=params['distillGamma']
    rtol=params['distillRtol']
    atol=params['distillAtol']

    velocity=Velocity().to(device)
    velocityAnswer=Velocity().to(device)

    
    vOptimizer=optim.AdamW(velocity.parameters(),lr=params['distillLearnRateVelocity'],
                           betas=betas,weight_decay=weightDecay)
     
    if os.path.exists(params['distillModelPath']):
        all=torch.load(params['distillModelPath'])
        velocityAnswer.load_state_dict(all['velocity'],strict=False)
    else:
        raise Exception('Your model path to be distilled doesn\'t exist.')
        
    if os.path.exists(params['distillCheckPointPath']):
        
        all=torch.load(params['distillCheckPointPath'])
        velocity.load_state_dict(all['velocity'])
        vOptimizer.load_state_dict(all['vOptimizer'])
 
        
        nowStep=all['step']
        nowEpoch=all['epoch']
        
        
        for param_group in vOptimizer.param_groups:
            param_group['betas']=betas

        for param_group in vOptimizer.param_groups:
            param_group['weight_decay']=weightDecay
        
    else:
        nowStep=0
        nowEpoch=0
        all=torch.load(params['distillModelPath'])
        velocity.load_state_dict(all['velocity'],strict=True)
        path=params['distillCheckPointPath']
            
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
                    'distilled':True,
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
    
    while True:
        NFE=None
        tqdmLoader=tqdm(trainLoader,desc=f'distill Epoch: {nowEpoch}, starting step={nowStep}')
        for (audios,mels) in tqdmLoader:
                
            with torch.cuda.amp.autocast(enabled=mixTraining):
                
                x1=audios.to(device)
                mels=mels.to(device)
               
                energy=(2**mels).sum(dim=1).sqrt().unsqueeze(1)
                sigma=(energy/maximunEnergy).repeat_interleave(repeats=params['hopSize'],dim=2)
                x0=sigma*torch.randn_like(sigma)
                
                if nowStep%5==0 or NFE==None:
                    melSaved=mels
                    x0Saved=x0
                    with torch.no_grad():
                        NFE=0
                        def f(t,x):
                            nonlocal NFE 
                            NFE+=1
                            return (velocityAnswer(x,mels,1-(-t).exp())-x)
                        
                        solution=odeint(f,x0,torch.tensor([0.0,5.0]).to(device),
                                    rtol=rtol,atol=atol,method='dopri5')[-1]
                    
    
                predict=velocity(x0,mels,torch.zeros(mels.size(0),1,1).to(device))
                fakeMels=(melProcessor(predict)+1).log2()
                realMels=(melProcessor(x1)+1).log2()
                MelLoss=(fakeMels-realMels).abs().mean()
                
                ODELoss=(predict-x1).pow(2).mean()
                L1Loss=(predict-x1).abs().mean()
                
                loss=ODELoss+0.015*MelLoss+0.025*L1Loss
                
                vOptimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(vOptimizer)
                scaler.step(vOptimizer)            
                scaler.update()       
            
                tqdmLoader.set_postfix(ODE=round(ODELoss.sqrt().item(),4),
                                    L1Loss=round(L1Loss.item(),4),
                                    MelLoss=round(MelLoss.item(),4),
                                    NFE=NFE)
                 
        
                predict=velocity(x0Saved,melSaved,torch.zeros(mels.size(0),1,1).to(device))
                fakeMels=(melProcessor(predict)+1).log2()
                realMels=(melProcessor(solution)+1).log2()
                MelLoss=(fakeMels-realMels).abs().mean()
                
                ODELoss=(predict-solution).pow(2).mean()
                L1Loss=(predict-solution).abs().mean()
                
                loss=ODELoss+0.015*MelLoss+0.025*L1Loss
                
                vOptimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(vOptimizer)
                scaler.step(vOptimizer)            
                scaler.update()       
                
                tqdmLoader.set_postfix(ODE=round(ODELoss.sqrt().item(),4),
                                       L1Loss=round(L1Loss.item(),4),
                                       MelLoss=round(MelLoss.item(),4),
                                       NFE=NFE)
                  
                nowStep+=1
                
                if nowStep%params['distillCheckPointSavingStep']==0:
                
                    path=params['distillCheckPointPath']
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
                                'distilled':True,
                                },path)
                
                if nowStep%params['distillLearnRateDecayStep']==0:
                    for param_group in vOptimizer.param_groups:
                        param_group['lr']*=gamma
                        
                if nowStep>=params['distillSteps']:
                    return    
                       
        nowEpoch+=1
            

if __name__=='__main__':
    distillation()
    
