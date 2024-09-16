params = dict(
    # params for mel-spectrograms
    fftSize=1024,
    windowSize=1024,
    hopSize=256,  # don't change STFT's hopsize
    melBands=100,
    sampleRate=24000,
    fmin=0,
    fmax=12000,
    melTrainWindow=64,  # The shape of training audios = batch * 1 * (melTrainWindow*hopSize)

    # params for training
    trainWithHybridPrecision=False,
    trainLearnRateVelocity=1e-4,
    trainGamma=0.997,
    trainLearnRateDecayStep=1000,
    trainBetas=(0.9, 0.99),
    trainWeightDecay=0.005,
    trainSteps=1000000,
    trainBatch=16,
    trainCheckPointSavingStep=10000,

    trainAudiosPath="./LibriTTS/train",
    trainMelsPath="./trainMels",
    trainCheckPointPath="./checkpoints/waveODE_0",
    trainDevice="cuda:0",

    # params for distillation (training hyperparameters and data are kept)
    distillWithHybridPrecision=False,
    distillLearnRateVelocity=2e-5,
    distillBetas=(0.8, 0.98),
    distillWeightDecay=0.05,
    distillGamma=0.98,
    distillLearnRateDecayStep=1000,
    distillDeltaT=0.005,
    distillDeltaTDecayRate=0.5,
    distillDeltaTDecayStep=400000,
    distillSteps=100000,
    distillBatch=16,
    distillCheckPointSavingStep=10000,

    distillModelPath="./checkpoints/waveODE_1000000",
    distillCheckPointPath="./distillation/distilledWaveODE_0",
    distillAudiosPath="./LibriTTS/train",
    distillMelsPath="./trainMels",
    distillDevice="cuda:0",

    # params for inference
    inferenceSteps=6,

    inferenceMelsPath="./libriTestMel0.01",
    inferenceSavingPath="./generation",
    inferenceCheckPointPath=
    # './checkpoints/waveODE_1000000',
    "./distillation/distilledWaveODE_30000",
    inferenceDevice="cuda:0",

    # params for model
    timeEmbeddingSize=512,
    velocityChannels=[512, 256, 128, 64, 32],
    velocityUpSampleRates=[8, 8, 2, 2],
    velocityKernelSizesUp=[[3, 7, 11], [3, 7, 11], [3, 7, 11], [3, 7, 11]],
    velocityDilationsUp=[[1, 3, 5], [1, 3, 5], [1, 3, 5], [1, 3, 5]],
    velocityKernelSizesDown=[[3], [3], [3], [3]],
    velocityDilationsDown=[[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
)
