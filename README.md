# L2S: Learning to Sample

A Matlab implementation of L2S: Learning to Sample. L2S identifies the intrinsic dimensions, constructed by influential configuration options, of a highly-configurable system to identify appropriate points to sample in the target environment. The samples are generated based on the model that has been learned in the source environment. L2S also transfers the model that has been learned int he source to the target to start off learning with an initial model.

## Demo

First set the correct path to appropriate datasets. Then run the following script in Matlab:
```matlab
setup
l2s
```

## Paper
```text
Jamshidi, Pooyan, Miguel Velez, Christian Kästner, and Norbert Siegmund. "Learning to Sample: Exploiting Similarities Across Environments to Learn Performance Models for Configurable Systems." FSE, 2018.
```