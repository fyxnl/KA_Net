# KA_Net
Real-World Haze Removal

> **Abstract:** *Restoring high-quality images from degraded hazy observations is a fundamental and essential task in the field of computer
vision. While deep models have achieved significant success with synthetic data, their effectiveness in real-world scenarios remains
uncertain. To improve adaptability in real-world environments, we construct an entirely new computational framework by making efforts
from three key aspects: imaging perspective, structural modules, and training strategies. To simulate the often-overlooked multiple
degradation attributes found in real-world hazy images, we develop a new hazy imaging model that encapsulates multiple degraded
factors, assisting in bridging the domain gap between synthetic and real-world image spaces. In contrast to existing approaches that
primarily address the inverse imaging process, we design a new dehazing network following the “localization-and-removal” pipeline.
The degradation localization module aims to assist in network capture discriminative haze-related feature information, and the
degradation removal module focuses on eliminating dependencies between features by learning a weighting matrix of training samples,
thereby avoiding spurious correlations of extracted features in existing deep methods. We also define a new Gaussian perceptual
contrastive loss to further constrain the network to update in the direction of the natural dehazing. Regarding multiple full/no-reference
image quality indicators and subjective visual effects on challenging RTTS, URHI, and Fattal real hazy datasets, the proposed method
has superior performance and is better than the current state-of-the-art methods.*
<hr />

## Requirement

- Python 3.7
- Pytorch 1.7.0

## Test
* Place the pre-training weight in the `model_dir` folder.
* Place test low-visibility images in the `test_dir` folder.
* Modify the weight name in the `feng_testniu_up.py`.<br>
* Run `feng_testniu_up.py`
* The results is saved in `expname` folder.

## Citation
