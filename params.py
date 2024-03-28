params=dict(
    #params for mel-spectrograms 
    fftSize=1024,
    windowSize=1024, 
    hopSize=256, # don't change STFT's hopsize
    melBands=100,
    sampleRate=24000,
    fmin=0,
    fmax=12000,
    melTrainWindow=62, # The shape of training audios = batch * 1 * (melTrainWindow*hopSize)
    
    #params for training
    trainWithHybridPrecision=False,
    trainLearnRateVelocity=1e-4,
    trainLearnRateMixer=1e-4,
    trainGamma=0.997,
    trainLearnRateDecayStep=8500,
    trainBetas=(0.9,0.99),
    trainWeightDecay=0.0005,
    
    trainAudiosPath='C:/deep_learning/LibriTTS/train',
    trainMelsPath='S:/waveODE/trainMels',
    trainCheckPointPath='./checkpoints/waveODE_0',
    
    trainSteps=1000000,
    trainBatch=16,
    trainCheckPointSavingStep=10000,
    trainDevice='cuda:0',
    
    #params for distillation (training hyperparameters and data are kept)
    distillWithHybridPrecision=False,
    distillModelPath='./checkpoints/waveODE_1000000',
    distillCheckPointPath='./distillation/distilledWaveODE_0',
    
    distillAudiosPath='C:/deep_learning/LibriTTS/train',
    distillMelsPath='S:/waveODE/trainMels',
    
    distillLearnRateVelocity=2e-5,
    distillBetas=(0.9,0.99),
    distillWeightDecay=0.0005,
    distillGamma=0.98,
    distillLearnRateDecayStep=1000,
    
    distillSteps=100000,
    distillBatch=16,
    distillCheckPointSavingStep=500,
    
    distillAtol=1e-5,
    distillRtol=1e-5,
    distillDevice='cuda:0',
    
    #params for inference
    inferenceWithHybridPrecision=False,
    inferenceMelsPath='S:/waveODE/libriTestMel0.01',
    inferenceSavingPath='./inference',
    inferenceCheckPointPath=
    #'./checkpoints/waveODE_1000000',
    './distillation/distilledWaveODE_44500',
    inferenceMethod='dopri5',
    inferenceAtol=1e-5,
    inferenceRtol=1e-5,
    inferenceDevice='cuda:0',
    
    #params for model
    timeEmbeddingSize=512,
    velocityChannels=[512,256,128,64,32],
    velocityUpSampleRates=[8,8,2,2],
    velocityKernelSizesUp=[[3,7,11],[3,7,11],[3,7,11],[3,7,11]],
    velocityDilationsUp=[[1,3,5],[1,3,5],[1,3,5],[1,3,5]],       
    velocityKernelSizesDown=[[3],[3],[3],[3]],
    velocityDilationsDown=[[1,1,1],[1,1,1],[1,1,1],[1,1,1]],  
  
)