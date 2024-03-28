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

import time
from torchdiffeq import odeint


def inference(mixTraining=params['inferenceWithHybridPrecision']):
    
    with torch.no_grad(): 

        device=params['inferenceDevice']
        method=params['inferenceMethod']
        rtol=params['inferenceRtol']
        atol=params['inferenceAtol']
        sampleRate=params['sampleRate']
        
        velocity=Velocity().to(device)
        
        distilled=False
        if os.path.exists(params['inferenceCheckPointPath']):
            
            all=torch.load(params['inferenceCheckPointPath'])
            velocity.load_state_dict(all['velocity'],strict=False)
            
            nowStep=all['step']
            print(f'{nowStep} steps model is loaded.')
            print(f'Params: {sum([param.numel() for param in velocity.parameters()])/1e6}M')
            if all.get('distilled')!=None:
                distilled=True
                print(f'The model is distilled.')
            else:
                print('The model is not distilled.')
            
        else:
            raise Exception('Your checkpoint path doesn\'t exist.')            

        maximunEnergy=torch.sqrt(torch.tensor(params['melBands']*32768.0))
        melPath=params['inferenceMelsPath']
        savingPath=params['inferenceSavingPath']
        allFiles=os.listdir(melPath)
        files=[name for name in allFiles if name.endswith('.mel')]
        loader=tqdm(files,desc='Inference ')
        inferenceTime=0
        audioTime=0
        NFE=0
        amount=0
        for name in loader:
            amount+=1
            melSpectrogram=torch.load(melPath+'/'+name).unsqueeze(0).to(device)
            start=time.time()
            energy=(2**melSpectrogram).sum(dim=1).sqrt().unsqueeze(1)
            sigma=(energy/maximunEnergy).repeat_interleave(repeats=params['hopSize'],dim=2)
            x0=sigma*torch.randn_like(sigma)
          
            if distilled==True:
                predict=velocity(x0,melSpectrogram,torch.zeros(1,1).to(device))
                
            else:
               
                def f(t,x):
                    nonlocal NFE 
                    NFE+=1         
                    return velocity(x,melSpectrogram,1-(-t).exp())-x

                predict=odeint(f,x0,torch.tensor([0.0,5.0]).to(device),
                                rtol=rtol,atol=atol,method=method)[-1]

        
            end=time.time()
            inferenceTime+=end-start
            audioTime+=melSpectrogram.size(-1)*256.0/sampleRate
            torchaudio.save(savingPath+'/'+name[:-4]+'.wav',
                            predict[0].cpu(),sampleRate)
            
            loader.set_postfix(NFE=round(NFE/amount,2),
                               AudioTime=round(audioTime,2),
                               InferenceTime=round(inferenceTime,2),
                               RTF=round(audioTime/inferenceTime,2))

if __name__=='__main__':
    inference()