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
from torchdiffeq import odeint


def stft_loss(answer,predict,
              fft_sizes=[1024, 2048, 512],
              hop_sizes=[120, 240, 50],
              win_lengths=[600, 1200, 240],
              window=torch.hann_window):
    loss=0
    for i in range(len(fft_sizes)):
        answerStft=torch.view_as_real(
            torch.stft(answer.squeeze(1),n_fft=fft_sizes[i],hop_length=hop_sizes[i],
                       win_length=win_lengths[i],window=window(win_lengths[i],device=answer.device),
                       return_complex=True))
        predictStft=torch.view_as_real(
            torch.stft(predict.squeeze(1),n_fft=fft_sizes[i],hop_length=hop_sizes[i],
                       win_length=win_lengths[i],window=window(win_lengths[i],device=predict.device),
                       return_complex=True))
        
        answerRealStft=torch.sqrt(answerStft[...,0]**2+answerStft[...,1]**2+1e-4)
        predictRealStft=torch.sqrt(predictStft[...,0]**2+predictStft[...,1]**2+1e-4)
    
        loss+=(answerRealStft-predictRealStft).norm(p='fro')/answerRealStft.norm(p='fro')
        loss+=(answerRealStft.log2()-predictRealStft.log2()).abs().mean()
    return loss/len(fft_sizes)


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
    velocity.train()
    velocityAnswer.eval()
    meanMelLoss=None
    meanSTFTLoss=None
    meanVelocityL1Loss=None
    meanVelocityL2Loss=None
    while True:
        NFE=None
        tqdmLoader=tqdm(trainLoader,desc=f'distill Epoch: {nowEpoch}, starting step={nowStep}')
        for (audios,mels) in tqdmLoader:
            
            
            with torch.cuda.amp.autocast(enabled=mixTraining):
                
                x1=audios.to(device)
                mels=mels.to(device)
                energy=(2**mels).sum(dim=1).sqrt().unsqueeze(1)
                sigma=F.interpolate((energy/maximunEnergy).clamp(min=0.001),size=(energy.size(-1)*params['hopSize']))
                x0=sigma*torch.randn_like(sigma)
                
                if nowStep%4==0 or NFE==None:
                    melSaved=mels
                    x0Saved=x0
                    with torch.no_grad():
                        NFE=0
                        def f(t,x):
                            nonlocal NFE 
                            NFE+=1
                            return (velocityAnswer(x,mels,1-(-t).exp())-x)
                            
                        x1Saved=odeint(f,x0,torch.tensor([0.0,5.0]).to(device),
                                    rtol=rtol,atol=atol,method='dopri5')[-1]
                            
                t=torch.zeros(x0.size(0),1,1).to(device)
                predict=velocity(x0,mels,t)
                delta=(predict-x1)
                fakeMels=(melProcessor(predict)+1).log2()
                realMels=(melProcessor(x1)+1).log2()
                
                melLoss=(fakeMels-realMels).abs().mean()
                STFTLoss=stft_loss(x1,predict)
                velocityL1Loss=delta.abs().mean()
                velocityL2Loss=(delta.pow(2)*(1.0/(1-t).clamp(min=0.1))).mean()
                loss=velocityL2Loss+0.01*melLoss+0.02*velocityL1Loss+0.005*STFTLoss
                
                vOptimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(vOptimizer)
                scaler.step(vOptimizer)
                scaler.update()   
                
                t=torch.zeros(x0Saved.size(0),1,1).to(device)
                predict=velocity(x0Saved,melSaved,t)
                delta=(predict-x1Saved)
                fakeMels=(melProcessor(predict)+1).log2()
                realMels=(melProcessor(x1Saved)+1).log2()
                
                melLoss=(fakeMels-realMels).abs().mean()
                STFTLoss=stft_loss(x1Saved,predict)
                velocityL1Loss=delta.abs().mean()
                velocityL2Loss=(delta.pow(2)*(1.0/(1-t).clamp(min=0.1))).mean()
                loss=velocityL2Loss+0.01*melLoss+0.02*velocityL1Loss+0.005*STFTLoss   
                vOptimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(vOptimizer)
                scaler.step(vOptimizer)
                scaler.update()   
                
                if meanMelLoss==None:
                    meanMelLoss=melLoss.item()
                    meanVelocityL1Loss=velocityL1Loss.item()
                    meanVelocityL2Loss=velocityL2Loss.sqrt().item()
                    meanSTFTLoss=STFTLoss.sqrt().item()   
                else:
                    meanMelLoss=meanMelLoss*0.99+0.01*melLoss.item()
                    meanVelocityL1Loss=meanVelocityL1Loss*0.99+0.01*velocityL1Loss.item()
                    meanVelocityL2Loss=meanVelocityL2Loss*0.99+0.01*velocityL2Loss.sqrt().item()
                    meanSTFTLoss=meanSTFTLoss*0.99+0.01*STFTLoss.sqrt().item()
                
                tqdmLoader.set_postfix(L1Loss=round(meanVelocityL1Loss,4),
                                       L2Loss=round(meanVelocityL2Loss,4),
                                       MelLoss=round(meanMelLoss,4),
                                       STFTLoss=round(meanSTFTLoss,4),
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
    
