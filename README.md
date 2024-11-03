# Project Overview

## Project Structure

```plaintext
├── figures (Contains image files generated by the simulations or analyses.)
│   ├── Nov2
│   ├── Oct7
│   ├── Oct8
│   └── Oct28
├── GaussianUniver (the Hermite kernel feature model simulation package.)
│   ├── __init__.py
│   ├── ERM.py
│   ├── ERMleaveOneOut.py
│   ├── hermiteFeature.py
│   ├── Optimization.py
│   ├── symmetricTensorNorm.py
│   └── test.py
├── GaussianUniverRandomFeature (the random feature model somulation package)
│   ├── __init__.py
│   ├── ERM.py
│   ├── hermiteFeature.py
│   └── test.py
├── previousCode (includes older files that are not currently relevant to the main simulations)
├── RandomFeatureTest.log (history)
├── operator_norm_s_repeat8.png
├── output.png
├── ERMcontrol.py (the ERM problem associated with the Hermite kernel feature model)
├── HemiteNorm.py
├── packageTest.py
└── RandomFeatureTest.py (the ERM problem within the random feature model framework)
```

## Detailed Documentation

For more in-depth documentation on each package, please refer to the respective `README.md` files located in each directory. These files provide detailed explanations and usage instructions for the scripts and modules within each package.

- [GaussianUniver README](./GaussianUniver/README.md)
- [GaussianUniverRandomFeature README](./GaussianUniverRandomFeature/README.md)

Note, in our codes the labels y are independent of x.
