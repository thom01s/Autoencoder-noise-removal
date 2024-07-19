# Autoencoder noise removal
> Usage of simple autoencoder to remove noise in cifar-10 images

 Autoencoder have the capacity of reducing informations of samples at the same time it recognizes the most important features of them.

 Using autoencoders it is possible to deconstruct images to its most basic form and them reconstruct them , trying to mantaing the same level of detalis at the output.The model compares the outputs with the inputs of the model to adjust the neuron weigths. 
 
 In the case of noise removal, we utilize a preprocessed image that has noise added and the output is compared directly with the original image(without noise). That way, the model learns that the noise doesn't represent any important features and stop reproducing them in the output.

 This kind of model can be used both for images as showed by the project or sounds and other timeseries, to remove background noises or interference in the recording devices.

## Development setup

 To use the model it is necessary to install some dependencies if they're not already intalled, such as:

```sh
keras
numpy
matplotlib
```

## Meta

Thomas Sponchiado Pastore – [@ThomasSPastore](https://twitter.com/ThomasSPastore) – thomas.spastore@gmail.com

[https://github.com/thom01s](https://github.com/thom01s?tab=repositories/)

## Contributing

1. Fork it (<https://github.com/thom01s/Autoencoder-noise-removal/fork>)
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

<!-- Markdown link & img dfn's -->
[npm-image]: https://img.shields.io/npm/v/datadog-metrics.svg?style=flat-square
[npm-url]: https://npmjs.org/package/datadog-metrics
[npm-downloads]: https://img.shields.io/npm/dm/datadog-metrics.svg?style=flat-square
[travis-image]: https://img.shields.io/travis/dbader/node-datadog-metrics/master.svg?style=flat-square
[travis-url]: https://travis-ci.org/dbader/node-datadog-metrics
[wiki]: https://github.com/yourname/yourproject/wiki
