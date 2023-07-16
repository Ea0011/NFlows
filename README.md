# Normalizing Flows

This repo contains pytorch implementations of several popular normalizing flows. Normalizing flows are a type of generative models that allow exact density computation through change of variables formula. The whole process involves sampling a simple random variable (eg. Gaussian), and transforming it to a sample from a complex distribution using invertible transformations. For computing the density of the dataset, the inverse flow is used to map a sample to a simple distribution and computing the probability. For sampling, a simple example is drawn and transformed to a sample from the target dataset. 

Here are some examples of normaliing flows learning a density of the given dataset.

<div align="center">
  <img src="https://github.com/Ea0011/NFlows/blob/main/demo/glow.gif" height="400" width="45%" />
  <img src="https://github.com/Ea0011/NFlows/blob/main/demo/radial.gif" height="400" width="45%" />
  <p>
    <em>Left: Glow learning a mixture of 6 Gaussians. Right: Radial flow learning a mixture of 3 Gaussians</em>
  </p>
</div>
