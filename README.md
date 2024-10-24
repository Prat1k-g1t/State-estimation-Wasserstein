# State-estimation-Wasserstein
Inverse problem (state estimation) for measure-valued conservation laws. The problem is posed in the Wasserstein space (the space of probability measures with finite second moments). The project involves manifold learning (using Wasserstein barycenters), optimal sensor placement (using consensus-based optimization) and state reconstruction.

The developments are made on the code: https://gitlab.tue.nl/20220022/sinkhorn-rom
The ML libraries used are: Scikit-learn and PyTorch
The Wasserstein metric is approximated using the sinkhorn algorithm: https://www.kernel-operations.io/geomloss/

New additions:
1. Sensor placement strategy using Wasserstein calculus
2. Implementation of optimal sensor location using PyTorch SGD and Consensus-based Optimization
3. State reconstruction using Wasserstein barycenters
