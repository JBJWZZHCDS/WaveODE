import torch
import torch.nn as nn
import torch.nn.functional as F
from params import params
from modules import AntiAliasingSnake,ResLayer,Snake,UpSampler,DownSampler
from torch.nn.utils import weight_norm,remove_weight_norm
    
class Velocity(nn.Module):
    
    def timeEmbedding(self,t):
            
            if len(t.shape)==1:
                t=t.unsqueeze(-1)  # batch -> batch*1    
            if len(t.shape)==3:
                t=t.squeeze(-1)    # batch*1*1 -> batch*1
                
            pos=torch.arange(64,device=t.device).unsqueeze(0) #1*64    
            table=100*t*10.0**(pos*4.0/63.0)  #batch*64 
            
            return torch.cat([torch.sin(table), torch.cos(table)], dim=1) #batch*128
        
    def __init__(self,channels=params['velocityChannels'],
                 upSampleRates=params['velocityUpSampleRates'],
                 kernelSizesUp=params['velocityKernelSizesUp'],
                 dilationsUp=params['velocityDilationsUp'],
                 kernelSizesDown=params['velocityKernelSizesDown'],
                 dilationsDown=params['velocityDilationsDown']):
        super().__init__()
    
        self.timePre0=nn.Linear(128,params['timeEmbeddingSize'])
        self.timePre1=nn.Linear(params['timeEmbeddingSize'],params['timeEmbeddingSize'])
        self.SiLU=nn.SiLU()
        self.upSampleRates=upSampleRates
        
        size=7
        self.convUpIn=nn.Conv1d(params['melBands'],channels[0],size,1,padding='same')
        self.convDownIn=nn.Conv1d(1,channels[-1],size,padding='same')
        
        self.ups=nn.ModuleList()
        self.downs=nn.ModuleList()
        
        for i in range(len(upSampleRates)):

            self.ups.append(
                    nn.ConvTranspose1d(channels[i],channels[i+1],kernel_size=2*upSampleRates[i],
                                    stride=upSampleRates[i],padding=upSampleRates[i]//2),
                
                )#stride=2kernel=4padding
            
            self.downs.append(
                    nn.Conv1d(channels[i+1],channels[i],kernel_size=2*upSampleRates[i]+1,
                    stride=upSampleRates[i],padding=upSampleRates[i])
                )
            
        self.resLayerUps=nn.ModuleList()
        self.resLayerDowns=nn.ModuleList()
        self.timeDowns=nn.ModuleList()
        
        for i in range(len(upSampleRates)):

            self.timeDowns.append(nn.Linear(params['timeEmbeddingSize'],channels[i+1]))
            self.resLayerUps.append(ResLayer(channels[i+1],kernelSizesUp[i],dilationsUp[i]))
            self.resLayerDowns.append(ResLayer(channels[i+1],kernelSizesDown[i],dilationsDown[i]))    

                    
        self.convUpOut=nn.Conv1d(channels[-1],1,size,1,padding='same')
        self.actUpOut=Snake(channels=channels[-1])
 
    def applyWeightNorm(self):
        
        self.convDownIn=weight_norm(self.convDownIn)
        self.convUpIn=weight_norm(self.convUpIn)
        self.convUpOut=weight_norm(self.convUpOut)
        
        for i in range(len(self.resLayerUps)):
            self.resLayerUps[i].applyWeightNorm()
        for i in range(len(self.resLayerDowns)):
            self.resLayerDowns[i].applyWeightNorm()
        for i in range(len(self.ups)):
            self.ups[i]=weight_norm(self.ups[i])
        for i in range(len(self.downs)):
            self.downs[i]=weight_norm(self.downs[i])
     
    def removeWeightNorm(self):
        
        self.convDownIn=remove_weight_norm(self.convDownIn)
        self.convUpIn=remove_weight_norm(self.convUpIn)
        self.convUpOut=remove_weight_norm(self.convUpOut)
        
        for i in range(len(self.resLayerUps)):
            self.resLayerUps[i].removeWeightNorm()
        for i in range(len(self.resLayerDowns)):
            self.resLayerDowns[i].removeWeightNorm()
        for i in range(len(self.ups)):
            self.ups[i]=remove_weight_norm(self.ups[i])
        for i in range(len(self.downs)):
            self.downs[i]=remove_weight_norm(self.downs[i])
            
    def forward(self,x,melSpectrogram,t,k=1):
        
        timeEmbedding=self.timeEmbedding(t)
        timeEmbedding=self.SiLU(self.timePre0(timeEmbedding))
        timeEmbedding=self.SiLU(self.timePre1(timeEmbedding))
        
        x=self.convDownIn(x)
       
        skipConnections=[x.clone()]
        for i in range(len(self.downs)-1,-1,-1):
            x+=self.timeDowns[i](timeEmbedding).unsqueeze(-1)
            x=self.resLayerDowns[i](x)
            x=self.downs[i](x)
            
            skipConnections.append(x.clone())
            
        melSpectrogram=self.convUpIn(melSpectrogram)+k*skipConnections[-1]
              
        for i in range(len(self.ups)):
            
            melSpectrogram=self.ups[i](melSpectrogram)
            melSpectrogram+=k*skipConnections[-i-2]
            melSpectrogram=self.resLayerUps[i](melSpectrogram)
            
            
        out=self.actUpOut(melSpectrogram)
        out=self.convUpOut(out)
        out=torch.tanh(out)

        return out
 
    
class GeneratorDiffWaveForTesting(nn.Module):
    
    def __init__(self,channels=64,depth=30,mod=10):
        super().__init__()
        
        self.channels=channels
        self.melBands=params['melBands']
        
        self.timeFc1=nn.Linear(128,512)
        self.SiLU=nn.SiLU()
        self.timeFc2=nn.Linear(512,512)
        
        self.timeFcs=nn.ModuleList([])
        for i in range(depth):
            self.timeFcs.append(nn.Linear(512,channels))
        
        self.convMel1 = nn.ConvTranspose2d(1, 1, [3, 32], stride=[1, 16], padding=[1, 8])
        self.convMel2 = nn.ConvTranspose2d(1, 1,  [3, 32], stride=[1, 16], padding=[1, 8])
        self.LeakyReLU = nn.LeakyReLU(0.4)
        
        self.convIn=nn.Conv1d(1,channels,1)
        self.ReLU=nn.ReLU()
        
        self.convDils=nn.ModuleList([])
        self.convMels=nn.ModuleList([])
        self.convSplits=nn.ModuleList([])
        
        for i in range(depth):
            self.convDils.append(nn.Conv1d(channels,2*channels,3,padding=2**(i%mod),dilation=2**(i%mod)))
            self.convMels.append(nn.Conv1d(self.melBands,2*channels,1))
            self.convSplits.append(nn.Conv1d(channels,2*channels,1))
        
        self.convFinal1=nn.Conv1d(channels,channels,1)
        self.convFinal2=nn.Conv1d(channels,1,1)

         
    def timeEmbedding(self,t):
        
        if len(t.shape)==1:
            t=t.unsqueeze(-1)  # batch -> batch*1    
        if len(t.shape)==3:
            t=t.squeeze(-1)    # batch*1*1 -> batch*1
            
        pos=torch.arange(64,device=t.device).unsqueeze(0) #1*64    
        table=100*t*10.0**(pos*4.0/63.0)  #batch*64 
        
        return torch.cat([torch.sin(table), torch.cos(table)], dim=1) #batch*128   
    
    def forward(self,x,mel,time):
        time=self.timeEmbedding(time)
        time=self.timeFc1(time)
        time=self.SiLU(time)
        time=self.timeFc2(time)
        time=self.SiLU(time)
        
        mel=mel.unsqueeze(1)
        mel=self.convMel1(mel)
        mel=self.LeakyReLU(mel)
        mel=self.convMel2(mel)
        mel=self.LeakyReLU(mel)
        mel=mel.squeeze(1)
        
        x=self.convIn(x)
        x=self.ReLU(x)
        skips=[]
        for i in range(len(self.timeFcs)):
            timeNow=self.timeFcs[i](time).unsqueeze(-1)
            melNow=self.convMels[i](mel)
            res=x
            x=x+timeNow
            x=self.convDils[i](x)
            x+=melNow
            gate,signal=torch.chunk(x,2,dim=1)
            x=torch.sigmoid(gate)*torch.tanh(signal)
            x=self.convSplits[i](x)
            out,skip=torch.chunk(x,2,dim=1)
            x=(res+out)/torch.sqrt(torch.tensor(2.0))
            skips.append(skip)
        
        x=torch.stack(skips).sum(dim=0)/torch.sqrt(torch.tensor(len(skips)))
        x=self.convFinal1(x)
        x=self.ReLU(x)
        x=self.convFinal2(x)
        
        return x
