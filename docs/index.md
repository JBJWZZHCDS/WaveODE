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
  flex: 0.2;
  margin: 0px;
  min-width: 50px;
  max-width: 50px;
}
body {
    font-family: Arial, sans-serif;
    background-color: #f0f0f0;
    color: #333;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100vh;
    margin: 0;
}
.container {
    text-align: center;
    background: white;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
}
.button {
    background-color: #4CAF50;
    border: none;
    color: white;
    padding: 15px 32px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 16px;
    margin: 4px 2px;
    cursor: pointer;
    border-radius: 4px;
}
.button:hover {
    background-color: #45a049;
}
</style>

<div class="audio-container">
  <div class="container">
    <h3>Groundtruth</h3>
    <button class="button" onclick="playAudio()">Play Audio</button>
        <audio id="audio" src="0-Groundtruth.wav"></audio>
<!--     <audio controls>
      <source src="0-Groundtruth.wav" type="audio/wav">
      Your browser does not support the audio element.
    </audio> -->
  </div>
  <script>
        function playAudio() {
            var audio = document.getElementById('audio');
            audio.play();
        }
    </script>
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
