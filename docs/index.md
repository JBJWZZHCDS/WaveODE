---
layout: default
---

Probability flow-based models for image and audio synthesis, such as denoising diffusion probabilistic models and Poisson flow generative models, can be interpreted as modeling any ground truth distribution through the non-compressible fluid partial differential equation,  where the initial and final fluid density are the chosen prior and the ground truth distribution respectively. In this research, we analyse various previous models under the unified perspective of probability flow equation, and propose WaveODE, which is a reparameterized domain-specific rectified flow model for mel-spectrogram conditioned speech synthesis task. Since mel-spectrogram is a relatively strong condition which limits the possible audios to a small range, waveODE models the ground truth distribution with a mel-conditioned prior distribution rather than the standard Gaussian distribution, and adopts a distillation method to accelerate the inference process. Experimental results show that our model is comparable with previous vocoders in sample quality, and could generate waveforms within one step of inference.

# Model

![Model](./model_newnew.png)

Values in parentheses of Conv1d and ConvTranspose1d refer to (output channel, kernel width, dilation, padding). A layer takes same padding if the value for padding is omitted. In each ResBlock the channel size remains unchanged.

# Audio Samples

<style>
.audio-container {
  display: flex;
  justify-content: space-around;
  flex-wrap: wrap;
}

.audio-item {
  flex: 1;
  margin: 10px;
  min-width: 200px;
  max-width: 300px;
}
</style>

<div class="audio-container">
  <div class="audio-item">
    <h3>Groundtruth</h3>
    <audio controls>
      <source src="0-Diffwave.wav" type="audio/wav">
      Your browser does not support the audio element.
    </audio>
  </div>
  <div class="audio-item">
    <h3>WaveODE (Ours)</h3>
    <audio controls>
      <source src="0-Diffwave.wav" type="audio/wav">
      Your browser does not support the audio element.
    </audio>
  </div>
  <div class="audio-item">
    <h3>DiffWave</h3>
    <audio controls>
      <source src="0-Diffwave.wav" type="audio/wav">
      Your browser does not support the audio element.
    </audio>
  </div>
  <div class="audio-item">
    <h3>FastDiff</h3>
    <audio controls>
      <source src="0-FastDiff.wav" type="audio/wav">
      Your browser does not support the audio element.
    </audio>
  </div>
  <div class="audio-item">
    <h3>FreGrad</h3>
    <audio controls>
      <source src="0-FastDiff.wav" type="audio/wav">
      Your browser does not support the audio element.
    </audio>
  </div>
  <div class="audio-item">
    <h3>HifiGAN</h3>
    <audio controls>
      <source src="0-FastDiff.wav" type="audio/wav">
      Your browser does not support the audio element.
    </audio>
  </div>
  <div class="audio-item">
    <h3>PriorGrad</h3>
    <audio controls>
      <source src="0-FastDiff.wav" type="audio/wav">
      Your browser does not support the audio element.
    </audio>
  </div>
  <div class="audio-item">
    <h3>WaveGlow</h3>
    <audio controls>
      <source src="0-FastDiff.wav" type="audio/wav">
      Your browser does not support the audio element.
    </audio>
  </div>
</div>
