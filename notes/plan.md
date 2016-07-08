# Plan #
## Feature Engineering ##
Dima suggested that the most appropriate features to investigate would be topographic maps of slices of the power spectrum.  


## Deep Learning Architectures ##
The features we plan to start with are small cuts of frequency-space measures over time, interpolated in 2D space between the sensors.

The neural net architecture will be a variation on the idea of LSTMs over residual convolutional nets (convnets from now on).

The first step will be to just train on 2D convnets, without long short term memory recurrent neural networks (LSTM from now on). We will start with simple convnets, and quickly move on to more modern architectures, which are known to be pretty much strictly superior.

Concretely, we intend to experiment with different configurations of :
  - Width of the frequency bands
  - Type of CNN cells:
      - Residual convnet cells (resnets and resnet cells from now on) and Inception convnet cells have proven to be the best, learning more quickly and keeping on learning for a longer time, as their architecture allows for the gradients to flow more easily through the layers of the model (resnet contribution) & as the multi convolution per layer allows for safe model ensembling.
    - Try different network depths
    - Experimentations with dropout ratios vs batch-normalization

We will pretty quickly add a single layer LSTM to the top of the model. We will then experiment with:
  - Length of time clips
  - Number of samples per clip
  - Overlap of different clips
  - frame skipping in clips
  - depth of convnets when we have an LSTM on it

The second thing we intend to fit is raw signal ove
