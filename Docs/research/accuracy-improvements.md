# Improving Semantic-Segmentation Accuracy for Surgical Suturing Videos

**Date**: 2026-09-15
**Scope**: Fixed-camera surgical suturing videos; target classes pad/wound, needle, thread; noisy pseudo-labels; a few hundred frames; CPU-only training.
**Saved at**: `D:\REU\Docs\research\accuracy-improvements.md`

This report collects primary-source evidence on five levers that move accuracy in the stated regime: framework choice, preprocessing, augmentation, noise-tolerant training, and lessons from the surgical-segmentation challenge literature. A final section ranks the levers by expected impact for this project.

---

## 1. Segmentation frameworks for small datasets

### 1.1 Detectron2 (Facebook / Meta AI Research)

- **What it is**: A PyTorch-based platform for object detection, instance/panoptic/semantic segmentation. Semantic segmentation is provided via DeepLabV3(+) and related heads; instance segmentation via Mask R-CNN.
- **Repo**: <https://github.com/facebookresearch/detectron2>
- **Docs**: <https://detectron2.readthedocs.io/>
- **Model zoo**: <https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md> - includes COCO-pretrained Mask R-CNN, DeepLabV3+, and ImageNet-pretrained ResNet / ResNeXt backbones.
- **License**: Apache-2.0 ([LICENSE](https://github.com/facebookresearch/detectron2/blob/main/LICENSE)).
- **CPU training**: Explicitly supported via `MODEL.DEVICE cpu` ([Getting Started](https://detectron2.readthedocs.io/en/stable/tutorials/getting_started.html)). Single-GPU configs are documented with linear LR scaling.
- **Small-data behavior**: No built-in few-shot pipeline for segmentation; users typically fine-tune COCO-pretrained Mask R-CNN / DeepLabV3+ heads. The model zoo Copy-Paste baselines are relevant for small-data augmentation.
- **Fit for this project**: Heavy framework; CPU training is slow for Mask R-CNN. Best used if you want a strong Mask R-CNN baseline for *instance* segmentation of needle/thread, or DeepLabV3+ for semantic segmentation of the wound pad.


### 1.2 MMSegmentation (OpenMMLab)

- **What it is**: A PyTorch semantic-segmentation toolbox with 40+ methods (FCN, PSPNet, DeepLabV3+, UPerNet, SegFormer, UNet, etc.) and a large model zoo.
- **Repo**: <https://github.com/open-mmlab/mmsegmentation>
- **Docs**: <https://mmsegmentation.readthedocs.io/en/latest/>
- **Model zoo**: [docs/en/model_zoo.md](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/model_zoo.md) - ImageNet-pretrained backbones and COCO/ADE20K/Cityscapes-trained heads.
- **License**: Apache-2.0 (OpenMMLab ecosystem license policy).
- **CPU training**: Supported via `device='cpu'` in `init_model`; training scripts accept CPU device.
- **Small-data behavior**: Config-driven; easy to swap backbones and heads. Supports many losses out of the box (cross-entropy, dice, focal, Lovasz, OHEM, Tversky) - useful for noisy / imbalanced labels.
- **Fit for this project**: The broadest menu of semantic-segmentation architectures in one repo. SegFormer-B0 or UNet with a ResNet-50 / MiT-B0 ImageNet-pretrained backbone is a strong starting point.

### 1.3 PyTorch-UNet (milesial)

- **What it is**: A clean, minimal PyTorch re-implementation of the original U-Net ([Ronneberger et al., MICCAI 2015](https://arxiv.org/abs/1505.04597)) for image semantic segmentation.
- **Repo**: <https://github.com/milesial/Pytorch-UNet>
- **Pretrained weights**: A Carvana-trained checkpoint is available via `torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=True)` ([releases](https://github.com/milesial/Pytorch-UNet/releases/tag/v3.0)). The Carvana domain (cars on road) is far from surgical video, so the weights are useful mainly as an initialization sanity check, not as a domain prior.
- **License**: GPL-3.0 (repo `LICENSE` file - **verify before deployment**; I did not fetch the raw file in this session). **Note**: GPL is copyleft - incompatible with proprietary deployment. Acceptable for academic research.
- **CPU training**: Trains on CPU out of the box; the model is small (~3 M params at scale 0.5).
- **Small-data behavior**: The original U-Net paper demonstrates good results on <100 images via heavy augmentation and skip-connection-driven feature reuse. The milesial repo trained from scratch on 5k images to a 0.988 Dice on Carvana; for a few hundred frames, fine-tuning from the Carvana checkpoint or training from scratch with strong augmentation are both realistic.
- **Fit for this project**: The lowest-friction baseline. Small enough to train on CPU in hours, not days. Best first model to try.


### 1.4 YOLO-seg (Ultralytics YOLOv8 / YOLO11 / YOLO26)

- **What it is**: Instance-segmentation models in the YOLO family. YOLOv8-seg, YOLO11-seg, and the newer YOLO26-seg are all available with COCO-pretrained weights at five scales (n, s, m, l, x).
- **Docs**:
  - YOLOv8: <https://docs.ultralytics.com/models/yolov8/>
  - YOLO11: <https://docs.ultralytics.com/models/yolo11/>
  - Segmentation task: <https://docs.ultralytics.com/tasks/segment/>
- **Pretrained weights**: COCO-pretrained `.pt` files auto-download on first use. YOLO11n-seg: 38.9 mAP box / 32.0 mAP mask at 640 px, 65.9 ms CPU ONNX, 2.9 M params ([YOLO11 model page](https://docs.ultralytics.com/models/yolo11/)).
- **License**: **AGPL-3.0** for community use; Enterprise license required for proprietary / commercial use ([Ultralytics Licensing](https://www.ultralytics.com/license)). **Important for this project**: AGPL-3.0 requires open-sourcing the entire derivative work. Fine for academic research; problematic if the pipeline is later embedded in a commercial product.
- **CPU training**: Explicitly supported (`device='cpu'` in `model.train()`). The nano models are small enough for CPU fine-tuning in reasonable time.
- **Small-data behavior**: Ultralytics supports custom datasets via a YAML descriptor. The built-in Copy-Paste and Mosaic augmentations are valuable for small datasets. However, YOLO-seg is *instance* segmentation, not semantic - it outputs per-object masks, not per-pixel class maps. Converting to semantic masks for pad/wound/needle/thread is straightforward but adds a post-processing step.
- **Fit for this project**: Excellent if you want instance-level needle and thread masks. Overkill if you only need a semantic wound-pad mask. The AGPL license must be acceptable for your deployment model.

### 1.5 SegFormer (NVIDIA / HKU / Nanjing Univ.)

- **What it is**: A hierarchical Transformer encoder (Mix Transformer, MiT) with a lightweight all-MLP decoder. No positional encoding, so it generalizes across resolutions. Published at NeurIPS 2021.
- **Paper**: Xie et al., `SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers,` NeurIPS 2021. [arXiv:2105.15203](https://arxiv.org/abs/2105.15203).
- **Official code**: <https://github.com/NVlabs/SegFormer>
- **Hugging Face**: `nvidia/segformer-b0-finetuned-ade-512-512` (3.75 M params, ADE20K-finetuned) and B0-B5 MiT checkpoints at <https://huggingface.co/nvidia>. Docs: <https://huggingface.co/docs/transformers/main/en/model_doc/segformer>.
- **Also in MMSegmentation**: SegFormer is a first-class citizen in MMSegmentation model zoo.
- **License**: NVIDIA official repo uses a non-standard license (check `NVlabs/SegFormer` LICENSE); Hugging Face checkpoints are released under the same terms. MMSegmentation implementation is Apache-2.0.
- **CPU training**: Possible but slow - Transformer attention is quadratic in sequence length. B0 at 512x512 is tractable; B3+ is not realistic on CPU for more than a few epochs.
- **Small-data behavior**: The ADE20K / Cityscapes pretrained heads are strong priors. Fine-tuning B0 on a few hundred frames with a new 3-class head (pad/wound, needle, thread) is the recommended recipe. The Hugging Face `Trainer` API and the [semantic-segmentation notebook](https://github.com/huggingface/notebooks/blob/main/examples/semantic_segmentation.ipynb) make this straightforward.
- **Fit for this project**: Best accuracy-per-parameter among the pure-semantic options. B0 is the right size for CPU. Use via MMSegmentation or Hugging Face Transformers.

### 1.6 Framework summary

| Framework | Task | Smallest pretrained | CPU-friendly | License | Best for this project |
|---|---|---|---|---|---|
| Detectron2 | Instance + semantic | COCO Mask R-CNN, DeepLabV3+ | Yes (slow) | Apache-2.0 | Mask R-CNN instance baseline |
| MMSegmentation | Semantic | 40+ methods, ImageNet backbones | Yes | Apache-2.0 | Broadest architecture menu |
| milesial PyTorch-UNet | Semantic | Carvana (domain-mismatched) | Yes, fast | GPL-3.0 (verify) | Fastest first baseline |
| Ultralytics YOLO-seg | Instance | COCO-pretrained nano-xlarge | Yes (nano) | AGPL-3.0 / Enterprise | Instance needle/thread masks |
| SegFormer (HF / MMSeg) | Semantic | ADE20K / Cityscapes B0-B5 | B0 only | NVIDIA / Apache-2.0 (MMSeg) | Best accuracy-per-param |


---

## 2. Preprocessing for surgical video

### 2.1 CLAHE (Contrast Limited Adaptive Histogram Equalization)

- **What it is**: Local histogram equalization with a contrast clip limit to suppress noise amplification. Operates on grayscale or per-channel.
- **OpenCV docs**: [`cv::CLAHE`](https://docs.opencv.org/4.8.0/d6/db6/classcv_1_1CLAHE.html). Python: `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))`.
- **Original algorithm**: Zuiderveld, `Contrast Limited Adaptive Histogram Equalization,` in *Graphics Gems IV*, pp. 474-485, 1994.
- **Surgical relevance**: Endoscopic and microscope images often have uneven illumination and low contrast in tissue regions. CLAHE is a standard preprocessing step in endoscopic-image pipelines. Apply to the L channel of LAB or to grayscale; do **not** apply naively to all three RGB channels (it shifts color balance).
- **Caveat**: CLAHE changes pixel intensities non-linearly. If you use a pretrained model (e.g., ImageNet-normalized), apply CLAHE *before* the model normalization and verify that the shift does not hurt transfer performance.

### 2.2 Illumination / color normalization

- **Reinhard color transfer**: Reinhard et al., `Color Transfer between Images,` *IEEE Computer Graphics and Applications*, 21(4), 2001. [DOI:10.1109/38.946629](https://doi.org/10.1109/38.946629). Aligns per-channel mean and std in lab space to a reference image. Simple, fast, and domain-agnostic.
- **Macenko stain normalization**: Macenko et al., `A Method for Normalizing Histology Slides for Quantitative Analysis,` *ISBI 2009*, pp. 1107-1110. [DOI:10.1109/ISBI.2009.5193250](https://doi.org/10.1109/ISBI.2009.5193250). [PDF](https://www.cs.unc.edu/~mn/sites/default/files/macenko2009.pdf). Designed for H&E histology - it separates stains in optical-density space using SVD of the pixel cloud.
- **Caveat for surgical video**: Macenko and Vahadane assume a linear mixture of a *small number of known stains* (typically H&E). Surgical video has a much richer appearance palette (metallic instruments, blood, fat, pad material, sutures) that does **not** decompose into two or three stains. Applying Macenko to surgical video is **not recommended** without significant modification. Reinhard-style global color transfer to a reference frame is safer and cheaper.
- **Practical recommendation**: Pick one well-lit reference frame; align every other frame LAB-channel mean/std to it. This removes most of the illumination drift from the fixed camera without distorting the appearance of instruments.

### 2.3 Background subtraction (fixed-camera advantage)

- **OpenCV docs**: [Background Subtraction tutorial](https://docs.opencv.org/5.0/tutorials/others/background_subtraction.html). Two built-in methods:
  - `cv2.createBackgroundSubtractorMOG2()` - Gaussian-mixture model. [Class docs](https://docs.opencv.org/5.0/main_modules/classcv_1_1BackgroundSubtractorMOG2.html). Based on Zivkovic 2004 / 2006.
  - `cv2.createBackgroundSubtractorKNN()` - K-nearest-neighbors background model.
- **Original papers**:
  - Zivkovic, `Improved Adaptive Gaussian Mixture Model for Background Subtraction,` *ICPR 2004*. [PDF](https://www.zoranz.net/Publications/gmm.pdf).
  - Zivkovic & van der Heijden, `Efficient Adaptive Density Estimation per Image Pixel for the Task of Background Subtraction,` *Pattern Recognition Letters*, 2006.
- **Fixed-camera advantage**: With a truly static camera, the background model converges quickly and stably. MOG2 / KNN can produce a clean foreground mask that isolates the wound pad, needle, thread, and instruments from the static background (table, drape, camera mount). This mask is useful as:
  - A preprocessing step (restrict training / inference to the foreground region).
  - An additional input channel or auxiliary loss for the segmentation model.
  - A way to detect camera-motion segments (exclude them from training).
- **Caveat**: MOG2 detects *all* foreground, including the surgeon hands and gloves. You still need the semantic model to distinguish pad / needle / thread from hands.

### 2.4 Frame registration / alignment

- **OpenCV ECC**: `cv2.findTransformECC` implements the Enhanced Correlation Coefficient algorithm of Evangelidis & Psarakis, `Parametric Image Alignment with Enhanced Correlation Coefficient Maximization,` *IEEE TPAMI*, 30(10), 2008. [DOI:10.1109/TPAMI.2008.113](https://doi.org/10.1109/TPAMI.2008.113). Supports translation, Euclidean, affine, and homography motion models.
- **Tutorial / walkthrough**: [LearnOpenCV - Image Alignment (ECC)](https://learnopencv.com/image-alignment-ecc-in-opencv-c-python/).
- **Phase correlation**: `cv2.phaseCorrelate` implements the Fourier-based translation finder. Useful for sub-pixel translation estimation between consecutive frames.
- **Fixed-camera relevance**: Even a `fixed` camera has sub-pixel drift from vibration, thermal expansion, or cable tension. Registering every frame to a reference (e.g., the first clean frame) before training removes this drift and makes the pseudo-labels more consistent across time. For a truly static camera, a single affine or Euclidean warp per video is usually sufficient.
- **Practical recipe**: Compute the ECC warp from each frame to a reference frame (using the wound-pad region as the alignment target, masked from instruments). Apply the inverse warp to both the frame and its pseudo-label.


---

## 3. Augmentation for small data

### 3.1 Geometric and photometric augmentations

Standard geometric augmentations (horizontal flip, random rotation +/-10-15 deg, random scale +/-10%, random crop) are essential for a few-hundred-frame dataset. Photometric augmentations (brightness, contrast, hue/saturation jitter, Gaussian blur, Gaussian noise) simulate illumination and focus variation.

- **Albumentations**: The de-facto standard augmentation library for segmentation in PyTorch.
  - **Repo**: <https://github.com/albumentations-team/albumentations>
  - **Docs**: <https://albumentations.ai/docs/>
  - **Semantic-segmentation guide**: <https://albumentations.ai/docs/3-basic-usage/semantic-segmentation/> - documents synchronized image+mask transforms, nearest-neighbor mask interpolation, and the `mask` argument to `Compose`.
  - **Key point**: Spatial transforms (flip, rotate, affine, elastic) are applied identically to image and mask. Pixel transforms (brightness, contrast, noise) are applied to the image only. Mask interpolation defaults to `cv2.INTER_NEAREST` to avoid creating invalid class IDs.

### 3.2 CutMix for segmentation

- **Paper**: Yun et al., `CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features,` *ICCV 2019*. [arXiv:1905.04899](https://arxiv.org/abs/1905.04899). [PDF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Yun_CutMix_Regularization_Strategy_to_Train_Strong_Classifiers_With_Localizable_Features_ICCV_2019_paper.pdf).
- **Idea**: Cut a rectangular patch from image B and paste it into image A; mix the labels proportionally to area. For segmentation, the label mixing is pixel-wise: the pasted region mask replaces the target mask in that rectangle.
- **Effect**: Forces the model to learn from partial object views and improves localization. Particularly useful when the dataset has few frames.
- **Albumentations**: Not built-in as a single transform, but easily composed from `A.CoarseDropout` + mask replacement, or via the `A.MixUp` transform (classification-only; for segmentation, implement the mask-aware version manually or use the `CopyAndPaste` transform below).

### 3.3 Copy-Paste for segmentation

- **Paper**: Ghiasi et al., `Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation,` *CVPR 2021*. [arXiv:2012.07177](https://arxiv.org/abs/2012.07177).
- **Idea**: Randomly paste object instances (with their masks) from one image onto another. The paper shows +2-3 mask AP on COCO and +3.6 mask AP on LVIS rare categories on top of strong baselines.
- **Albumentations**: First-class support via `A.CopyAndPaste` ([docs](https://albumentations.ai/docs/api-reference/albumentations/augmentations/mixing/transforms/#albumentations.augmentations.mixing.transforms.CopyAndPaste)). Also `A.Mosaic` for YOLO-style 4-image mosaics.
- **Relevance to this project**: Extremely high. With a few hundred frames, Copy-Paste effectively multiplies the number of needle / thread / wound-pad appearances. Paste a needle from frame A onto the wound pad of frame B; the mask composes cleanly. This is one of the highest-impact augmentations for small surgical datasets.
- **Practical config** (from Albumentations docs):
  ```python
  A.CopyAndPaste(
      scale_range=(0.4, 1.0),
      blend_mode='gaussian',
      blend_sigma_range=(1.0, 2.0),
      min_visibility_after_paste=0.1,
      p=0.7,
  )
  ```

### 3.4 Recommended augmentation stack

For a few-hundred-frame surgical dataset:

1. **Geometric**: `HorizontalFlip(p=0.5)`, `Affine(scale=(0.9,1.1), rotate=(-10,10), translate_percent=(-0.05,0.05), p=0.5)`, `RandomCrop` or `CenterCrop` to a fixed training size.
2. **Photometric**: `RandomBrightnessContrast(p=0.3)`, `HueSaturationValue(p=0.3)`, `GaussianBlur(p=0.2)`, `GaussNoise(p=0.1)`.
3. **Object-aware**: `CopyAndPaste(p=0.5)` for needle and thread instances.
4. **Regularization**: `CoarseDropout(num_holes_range=(1,3), hole_height_range=(0.05,0.15), hole_width_range=(0.05,0.15), fill_value=0, mask_fill_value=None, p=0.3)` - drops random rectangles from the image but leaves the mask intact (acts like Cutout / CutMix-lite).


---

## 4. Noise-tolerant training for imperfect labels

### 4.1 Cleanlab / confident learning

- **What it is**: An open-source library for detecting label errors and training robust models via *confident learning*.
- **Repo**: <https://github.com/cleanlab/cleanlab>
- **Docs**: <https://docs.cleanlab.ai/stable/>
- **Original paper**: Northcutt, Jiang & Chuang, `Confident Learning: Estimating Uncertainty in Dataset Labels,` *JAIR*, 70, 1373-1411, 2021. [arXiv:1911.00068](https://arxiv.org/abs/1911.00068). [JAIR](https://jair.org/index.php/jair/article/view/12125).
- **How it works**: Given a classifier out-of-sample predicted probabilities and the (noisy) given labels, confident learning estimates the joint distribution of noisy and true labels, identifies likely label errors, and prunes them. `CleanLearning` wraps any scikit-learn-compatible classifier and handles the cross-validation + pruning + retraining pipeline automatically.
- **Segmentation support**: Cleanlab has published on label errors in semantic segmentation data (Lad & Mueller, `Estimating label quality and errors in semantic segmentation data via any model,` *ICML 2023 Data-centric ML Workshop*). The `cleanlab.token_classification` and segmentation utilities can flag suspect pixels / frames.
- **Relevance to this project**: High. With pseudo-labels from an auto-annotation pipeline, cleanlab can identify the frames (or pixels) most likely to be mislabeled. Pruning these before training, or using them with reduced weight, is a direct lever on accuracy.
- **Practical recipe**: Train a first-pass model -> compute out-of-sample `pred_probs` on the training set -> run `CleanLearning.find_label_issues()` -> remove or down-weight flagged frames -> retrain.

### 4.2 Label smoothing

- **Paper**: Szegedy et al., `Rethinking the Inception Architecture for Computer Vision,` *CVPR 2016*. [arXiv:1512.00567](https://arxiv.org/abs/1512.00567). Section 7: `Model Regularization via Label Smoothing.`
- **Idea**: Replace the one-hot target distribution `q(k) = delta_{k,y}` with `q'(k) = (1 - epsilon) delta_{k,y} + epsilon / K`, where `K` is the number of classes and `epsilon` is a small smoothing parameter (typically 0.1).
- **Effect**: Prevents the model from becoming over-confident, acts as a regularizer, and improves generalization. In ImageNet, LSR improved top-1 error by ~0.2% absolute.
- **Relevance to noisy labels**: Label smoothing is explicitly listed as a noise-robust technique in the symmetric-cross-entropy paper (Wang et al., ICCV 2019). It reduces overfitting to any single label, including a wrong pseudo-label.
- **Implementation**: One line in PyTorch: `nn.CrossEntropyLoss(label_smoothing=0.1)`.

### 4.3 Robust loss functions

#### 4.3.1 Generalized Cross Entropy (GCE)

- **Paper**: Zhang & Sabuncu, `Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels,` *NeurIPS 2018*. [arXiv:1805.07836](https://arxiv.org/abs/1805.07836). [PDF](https://proceedings.neurips.cc/paper_files/paper/2018/file/f2925f97bc13ad2852a7a551802feea0-Paper.pdf).
- **Idea**: `L_q(f(x), e_j) = (1 - f_j(x)^q) / q` for `q in (0, 1]`. Recovers CE as `q -> 0` and MAE at `q = 1`. The truncated variant further improves robustness.
- **Effect**: Theoretically noise-tolerant (symmetric loss property). Practically, GCE trains faster than MAE and is more robust than CE under both closed-set and open-set noise.

#### 4.3.2 Symmetric Cross Entropy (SCE)

- **Paper**: Wang et al., `Symmetric Cross Entropy for Robust Learning with Noisy Labels,` *ICCV 2019*. [arXiv:1908.06112](https://arxiv.org/abs/1908.06112). [PDF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Symmetric_Cross_Entropy_for_Robust_Learning_With_Noisy_Labels_ICCV_2019_paper.pdf).
- **Idea**: `SCE = alpha * CE + beta * RCE`, where `RCE` (Reverse Cross Entropy) is the noise-tolerant term. The CE term drives convergence; the RCE term provides robustness.
- **Effect**: Outperforms GCE, Forward, Bootstrap, and D2L on CIFAR-10/100 with symmetric and asymmetric noise up to 80%.
- **Relevance**: Easy to implement (a few lines of PyTorch). A good drop-in replacement for `nn.CrossEntropyLoss` when pseudo-labels are noisy.

#### 4.3.3 Bootstrapping

- **Paper**: Reed et al., `Training Deep Neural Networks on Noisy Labels with Bootstrapping,` *ICLR Workshop, 2015*. [arXiv:1412.6596](https://arxiv.org/abs/1412.6596).
- **Idea**: Replace the hard target with a convex combination of the hard target and the model own running prediction. `Soft bootstrapping` uses the train-time prediction; `hard bootstrapping` uses the predicted class.
- **Effect**: Simple, no extra data needed. Works well as a regularizer when label noise is moderate.

### 4.4 Co-teaching

- **Paper**: Han et al., `Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels,` *NeurIPS 2018*. [arXiv:1804.06872](https://arxiv.org/abs/1804.06872). [PDF](https://arxiv.org/pdf/1804.06872).
- **Idea**: Train two networks simultaneously. Each network selects its small-loss instances (likely clean) and teaches them to the peer network. The `small loss` criterion exploits the memorization effect: DNNs fit clean labels before noisy ones.
- **Extension**: Co-teaching+ (Yu et al., *ICML 2019*, [arXiv:1901.04215](https://arxiv.org/abs/1901.04215)) adds an `update by disagreement` step to prevent the two networks from converging to a consensus.
- **Relevance**: Powerful when noise is high (>30%). For a few hundred frames with pseudo-labels, co-teaching can be combined with a small U-Net or SegFormer-B0 pair. The cost is 2x training time, which is acceptable on CPU for small models.

### 4.5 Pseudo-label refinement

- **Self-training**: Train a model -> generate pseudo-labels on unlabeled frames -> filter by confidence threshold -> retrain. Cleanlab `ActiveLab` (Goh & Mueller, *ICLR 2023 Workshop*) provides an active-learning variant that suggests which frames to re-label next.
- **Confidence thresholding**: Only trust pseudo-labels where the model softmax probability exceeds a threshold (e.g., 0.9). Discard or flag the rest for human review.
- **Consensus from multiple models**: Train 3-5 models with different seeds or architectures; keep a pseudo-label only if all models agree. This is a cheap ensemble filter.


---

## 5. Surgical tool segmentation literature

### 5.1 EndoVis 2015 - first robotic instrument segmentation sub-challenge

- **Site**: <https://endovissub-instrument.grand-challenge.org>
- **Context**: Part of the MICCAI 2015 Endoscopic Vision workshop. Provided robotic images with *automatically generated* annotations from robot forward kinematics.
- **Limitations** (from the 2017 challenge paper): Limited background variation, lack of complex motion, and inaccuracies in the annotation due to cable-driven joint offsets in the dVRK.
- **Lesson**: Auto-generated labels from kinematics are noisy - directly relevant to this project pseudo-label situation.

### 5.2 EndoVis 2017 - Robotic Instrument Segmentation

- **Paper**: Allan et al., `2017 Robotic Instrument Segmentation Challenge,` 2019. [arXiv:1902.06426](https://arxiv.org/abs/1902.06426).
- **Site**: <https://endovissub2017-roboticinstrumentsegmentation.grand-challenge.org/>
- **Dataset**: 10 sequences of abdominal porcine procedures on da Vinci Xi. 300 frames per sequence, sampled at 1 Hz. 8 sequences for training (225 frames each), 2 for testing. Labels manually created by Intuitive Surgical.
- **Tasks**: (1) Binary instrument segmentation, (2) instrument part segmentation (shaft / wrist / jaws), (3) instrument type classification.
- **Winning methods**: Modified U-Net architectures dominated. The challenge report notes that `the challenge was won by a team using a modified U-Net architecture` and that `the accuracy of the binary and parts based methods for many of the participants exceeded 0.7 mIoU.`
- **Lesson**: U-Net variants are the established baseline for surgical instrument segmentation. The dataset is small (1800 training frames) - comparable to this project.

### 5.3 EndoVis 2018 - Robotic Scene Segmentation

- **Paper**: Allan et al., `2018 Robotic Scene Segmentation Challenge.` [arXiv:2001.11190](https://arxiv.org/abs/2001.11190). [PDF](https://arxiv.org/pdf/2001.11190v3.pdf).
- **Site**: <https://endovissub2018-roboticscenesegmentation.grand-challenge.org/>
- **Dataset**: 19 sequences (15 train, 4 test), 300 frames each, from porcine nephrectomy procedures. Stereo pairs at 1280x1024.
- **Classes**: Instrument parts (shaft/wrist/jaws), drop-in ultrasound probe, **suturing needles**, **suturing thread**, suction-irrigation devices, surgical clips, kidney parenchyma, covered kidney, small intestine, background.
- **Key relevance**: This is the **only public challenge dataset that includes suturing needles and thread as explicit segmentation classes**. The winning methods again used U-Net variants, often pre-trained on EndoVis 2017 and fine-tuned with augmentations (scaling, rotation, flipping, brightness, contrast).
- **Lesson**: Pre-training on EndoVis 2017 (instruments) and fine-tuning on the target dataset is a proven transfer strategy. The 2018 dataset is the closest public analog to this project target classes.

### 5.4 CATARACTS - Challenge on Automatic Tool Annotation for cataRACT Surgery

- **2017 tool detection**: 50 cataract surgery videos with frame-level instrument presence annotations. Site: <https://cataracts.grand-challenge.org/>.
- **2020 semantic segmentation**: Luengo et al., `2020 CATARACTS Semantic Segmentation Challenge.` [arXiv:2110.10965](https://arxiv.org/abs/2110.10965). 4670 training images and 531 test images with pixel-wise semantic annotations of 36 classes (anatomy + instruments). Three sub-tasks: (I) anatomy + merged instruments, (II) anatomy + grouped instruments, (III) anatomy + instrument tips and handles.
- **Dataset**: <https://ieee-dataport.org/open-access/cataracts>
- **Lesson**: The CATARACTS dataset is much larger than EndoVis and covers a different surgical domain (ophthalmology). The class imbalance and the variety of instrument appearances are relevant lessons for this project.

### 5.5 EndoVis challenge overview page

- **Datasets & Publications**: <https://opencas.dkfz.de/endovis/datasetspublications/> - central index of all EndoVis sub-challenges, papers, and dataset links.

### 5.6 Transfer lessons for a small fixed-camera suturing setup

1. **U-Net is the established baseline**. Every EndoVis challenge has been won or placed by U-Net variants. Start here.
2. **Pre-training on a related surgical dataset helps**. Pre-training on EndoVis 2017 (instruments) or EndoVis 2018 (scene, including needle + thread) and fine-tuning on your data is the recommended transfer path.
3. **Augmentations that work**: scaling, rotation, flipping, brightness, contrast - the same geometric + photometric stack recommended in section 3.
4. **Small datasets are the norm**. EndoVis 2017 had ~1800 training frames; EndoVis 2018 had ~4500. A few hundred frames is small but not unprecedented.
5. **Pseudo-labels are a known problem**. The 2015 challenge kinematics-generated labels were noisy and inaccurate - the community moved to manual labels in 2017. This project pseudo-labels should be treated with the same skepticism and cleaned (via cleanlab or manual review) before final training.


---

## 6. Ranked by expected impact

For a few-hundred-frame, noisy-label, fixed-camera, CPU-only setup, the following techniques are ranked by expected accuracy improvement, grounded in the evidence above.

| Rank | Technique | Rationale |
|---|---|---|
| 1 | **Fine-tune a pretrained SegFormer-B0 or U-Net** (from ADE20K / Cityscapes / EndoVis) | The single largest lever. A pretrained backbone provides features that a from-scratch model on a few hundred frames cannot learn. SegFormer-B0 has 3.75 M params and is CPU-tractable; U-Net is even smaller. ([SegFormer paper](https://arxiv.org/abs/2105.15203), [EndoVis 2017/2018 winning methods](https://arxiv.org/abs/1902.06426)). |
| 2 | **Clean pseudo-labels with cleanlab / confident learning** | Directly attacks the noise in the auto-generated labels. Pruning or down-weighting mislabeled frames before training is a high-leverage, low-cost step. ([cleanlab](https://github.com/cleanlab/cleanlab), [Northcutt et al. JAIR 2021](https://arxiv.org/abs/1911.00068)). |
| 3 | **Copy-Paste augmentation** | Effectively multiplies the number of needle / thread / wound-pad appearances. Proven to give +2-3 mask AP on COCO and +3.6 on LVIS rare categories. Built into Albumentations. ([Ghiasi et al. CVPR 2021](https://arxiv.org/abs/2012.07177), [Albumentations docs](https://albumentations.ai/docs/api-reference/albumentations/augmentations/mixing/transforms/)). |
| 4 | **Background subtraction (MOG2 / KNN) as a preprocessing mask** | Exploits the fixed-camera constraint to isolate foreground. Restricting training and inference to the foreground region reduces background confusion and improves effective data density. ([OpenCV docs](https://docs.opencv.org/5.0/tutorials/others/background_subtraction.html), [Zivkovic ICPR 2004](https://www.zoranz.net/Publications/gmm.pdf)). |
| 5 | **Symmetric Cross Entropy or GCE loss** | Drop-in replacement for CE that is provably noise-tolerant. SCE outperforms GCE, Forward, and Bootstrap on CIFAR with up to 80% noise. Easy to implement. ([Wang et al. ICCV 2019](https://arxiv.org/abs/1908.06112), [Zhang & Sabuncu NeurIPS 2018](https://arxiv.org/abs/1805.07836)). |
| 6 | **Geometric + photometric augmentation stack** (Albumentations) | Essential for any small dataset. Horizontal flip, small rotation, scale jitter, brightness/contrast, Gaussian blur. Prevents overfitting to the few hundred frames. ([Albumentations](https://github.com/albumentations-team/albumentations)). |
| 7 | **Label smoothing (epsilon = 0.05-0.1)** | One-line regularizer that reduces over-confidence in noisy pseudo-labels. Proven on ImageNet; cheap to add. ([Szegedy et al. CVPR 2016](https://arxiv.org/abs/1512.00567)). |
| 8 | **Frame registration to a reference (ECC)** | Removes sub-pixel camera drift, making pseudo-labels more consistent across time. Important if the camera is not perfectly rigid. ([OpenCV ECC](https://docs.opencv.org/4.4.0/dd/d93/samples%5F2cpp%5F2image%5Falignment%5F8cpp-example.html), [Evangelidis & Psarakis TPAMI 2008](https://doi.org/10.1109/TPAMI.2008.113)). |
| 9 | **CLAHE on the L channel (LAB)** | Improves contrast in tissue regions. Apply before model normalization. Do not apply to all RGB channels. ([OpenCV CLAHE](https://docs.opencv.org/4.8.0/d6/db6/classcv_1_1CLAHE.html), [Zuiderveld 1994](https://doi.org/10.1016/B978-0-08-050755-2.50050-8)). |
| 10 | **Co-teaching (dual-network training)** | Powerful when noise is very high (>30%). Costs 2x training time. Use if cleanlab + robust losses are not enough. ([Han et al. NeurIPS 2018](https://arxiv.org/abs/1804.06872)). |
| 11 | **Reinhard color normalization to a reference frame** | Removes illumination drift. Cheaper and safer than Macenko for surgical video. ([Reinhard et al. 2001](https://doi.org/10.1109/38.946629)). |

---

## Claims I could NOT verify against a primary source

- **MMSegmentation exact license file**: I confirmed the repo is Apache-2.0 from multiple secondary sources and the OpenMMLab ecosystem license policy, but I did not fetch the raw `LICENSE` file from the repo in this session.
- **milesial PyTorch-UNet license**: I stated GPL-3.0 based on general knowledge of the repo; I did not fetch the raw `LICENSE` file in this session. Verify at <https://github.com/milesial/Pytorch-UNet/blob/master/LICENSE> before deployment.
- **SegFormer B0 CPU inference time**: I did not find a primary-source benchmark for B0 on a specific CPU. The NeurIPS paper reports GPU FPS; CPU time will depend on hardware.
- **Cleanlab segmentation utilities**: The ICML 2023 workshop paper (Lad & Mueller) is cited in the cleanlab README, but I did not verify the exact API surface for segmentation in the current cleanlab release.
- **EndoVis 2015 winning method**: The 2017 challenge paper mentions the 2015 challenge limitations but does not name a specific winning team or method. I could not find a primary-source citation for the 2015 winner.
- **Zuiderveld 1994 DOI**: I cited the Graphics Gems IV chapter with a DOI pattern; the exact DOI should be verified against the publisher catalog.

