---
layout: default
---

Probability flow-based models for image and audio synthesis, such as denoising diffusion probabilistic models and Poisson flow generative models, can be interpreted as modeling any ground truth distribution through the non-compressible fluid partial differential equation,  where the initial and final fluid density are the chosen prior and the ground truth distribution respectively. In this research, we analyse various previous models under the unified perspective of probability flow equation, and propose WaveODE, which is a reparameterized domain-specific rectified flow model for mel-spectrogram conditioned speech synthesis task. Since mel-spectrogram is a relatively strong condition which limits the possible audios to a small range, waveODE models the ground truth distribution with a mel-conditioned prior distribution rather than the standard Gaussian distribution, and adopts a distillation method to accelerate the inference process. Experimental results show that our model is comparable with previous vocoders in sample quality, and could generate waveforms within one step of inference.

# Model

![Model](./model_newnew.png)

Values in parentheses of Conv1d and ConvTranspose1d refer to (output channel, kernel width, dilation, padding). A layer takes same padding if the value for padding is omitted. In each ResBlock the channel size remains unchanged.

# Audio Samples

<style>

.split-container {
    display: flex;
}

.left-pane, .right-pane {
    display: flex;
    align-items: center;
    justify-content: center;
}

.left-pane {
    flex: 1;
}

.right-pane {
    flex: 2;
}
.audio-item {
    flex: 1;
}
audio {
    width: 150px;
}
</style>


<div class="split-container">
    <div class="left-pane">
        <h2>Left Pane</h2>
        <p>This is the left pane content.</p>
    </div>
    <div class="right-pane">
        <h2>Right Pane</h2>
        <p>This is the right pane content.</p>
    </div>
</div>

<div class="grid-container">
    <div class="left-pane">
        <h2>Left Pane</h2>
        <p>This is the left pane content.</p>
    </div>
    <div class="right-pane">
        <div class="audio-container">
<!--   <h4>Groundtruth</h4> -->
  <div class="audio-item">
    <h3>Male Speech</h3>
    <audio controls>
      <source src="0-Groundtruth.wav" type="audio/wav">
      Your browser does not support the audio element.
    </audio>
  </div>
  <div class="audio-item">
    <h3>Drum</h3>
    <audio controls>
      <source src="1-Groundtruth.wav" type="audio/wav">
      Your browser does not support the audio element.
    </audio>
  </div>
  <div class="audio-item">
    <h3>Song</h3>
    <audio controls>
      <source src="2-Groundtruth.wav" type="audio/wav">
      Your browser does not support the audio element.
    </audio>
  </div>
  <div class="audio-item">
    <h3>Female Speech</h3>
    <audio controls>
      <source src="3-Groundtruth.wav" type="audio/wav">
      Your browser does not support the audio element.
    </audio>
  </div>
</div>
    </div>
</div>

<div class="audio-container">
<!--   <h4>Groundtruth</h4> -->
  <div class="audio-item">
    <h3>Male Speech</h3>
    <audio controls>
      <source src="0-Groundtruth.wav" type="audio/wav">
      Your browser does not support the audio element.
    </audio>
  </div>
  <div class="audio-item">
    <h3>Drum</h3>
    <audio controls>
      <source src="1-Groundtruth.wav" type="audio/wav">
      Your browser does not support the audio element.
    </audio>
  </div>
  <div class="audio-item">
    <h3>Song</h3>
    <audio controls>
      <source src="2-Groundtruth.wav" type="audio/wav">
      Your browser does not support the audio element.
    </audio>
  </div>
  <div class="audio-item">
    <h3>Female Speech</h3>
    <audio controls>
      <source src="3-Groundtruth.wav" type="audio/wav">
      Your browser does not support the audio element.
    </audio>
  </div>
</div>

<div class="audio-container">
  <h3>WaveODE (Ours)</h3>
  <div class="audio-item">
    <audio controls>
      <source src="0-WaveODE (Ours).wav" type="audio/wav">
      Your browser does not support the audio element.
    </audio>
  </div>
  <div class="audio-item">
    <audio controls>
      <source src="1-WaveODE (Ours).wav" type="audio/wav">
      Your browser does not support the audio element.
    </audio>
  </div>
  <div class="audio-item">
    <audio controls>
      <source src="2-WaveODE (Ours).wav" type="audio/wav">
      Your browser does not support the audio element.
    </audio>
  </div>
  <div class="audio-item">
    <audio controls>
      <source src="3-WaveODE (Ours).wav" type="audio/wav">
      Your browser does not support the audio element.
    </audio>
  </div>
</div>


<div class="audio-container">
  <div class="audio-item">
    <h3>WaveODE (Ours)</h3>
    <audio controls>
      <source src="0-WaveODE (Ours).wav" type="audio/wav">
      Your browser does not support the audio element.
    </audio>
  </div>
  <div class="audio-item">
    <h3>WaveODE (Ours)</h3>
    <audio controls>
      <source src="0-WaveODE (Ours).wav" type="audio/wav">
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
      <source src="0-FreGrad.wav" type="audio/wav">
      Your browser does not support the audio element.
    </audio>
  </div>
  <div class="audio-item">
    <h3>HifiGAN</h3>
    <audio controls>
      <source src="0-HifiGAN.wav" type="audio/wav">
      Your browser does not support the audio element.
    </audio>
  </div>
  <div class="audio-item">
    <h3>PriorGrad</h3>
    <audio controls>
      <source src="0-PriorGrad.wav" type="audio/wav">
      Your browser does not support the audio element.
    </audio>
  </div>
  <div class="audio-item">
    <h3>WaveGlow</h3>
    <audio controls>
      <source src="0-WaveGlow.wav" type="audio/wav">
      Your browser does not support the audio element.
    </audio>
  </div>
</div>
