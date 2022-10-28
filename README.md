# 3D_adversarial_attacks

# Papers
## 3D architectures
***Autoencoder
 
**[1] Point Transformer**
- intro: ICCV 2021, first work of point 3D with Transformer, learnable position embedding, transition down and up
- paper: [https://openaccess.thecvf.com/content/ICCV2021/papers/Zhao_Point_Transformer_ICCV_2021_paper.pdf](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhao_Point_Transformer_ICCV_2021_paper.pdf)
- github: [https://github.com/POSTECH-CVLab/point-transformer](https://github.com/POSTECH-CVLab/point-transformer)

**[2] PoinTr: Diverse Point Cloud Completion with Geometry-Aware Transformers**
- intro: ICCV 2021 Oral, tsing hua, Points Proxies(features of patches in images), Geometry-aware, Chamfer Distance (CD) and Earth
Mover’s Distance (EMD),solid
- arxiv: [https://arxiv.org/pdf/2108.08839.pdf](https://arxiv.org/pdf/2108.08839.pdf)
- code: [https://github.com/yuxumin/PoinTr](https://github.com/yuxumin/PoinTr)

**[3] PointASNL: Robust Point Clouds Processing using Nonlocal Neural Networks with Adaptive Sampling**
- intro: CVPR 2020, local non local, point cloud.
- arxiv: [https://arxiv.org/pdf/2003.00492.pdf](https://arxiv.org/pdf/2003.00492.pdf)
- code: [https://github.com/yanx27/PointASNL](https://github.com/yanx27/PointASNL)

***Loss terms

**[1] A Conditional Point Diffusion-Refinement Paradigm for 3D Point Cloud Completion**
- intro: ICLR 2022, Point Diffusion-Refinement, denoising diffusion probabilistic model (DDPM)
- paper: [https://openreview.net/forum?id=wqD6TfbYkrn](https://openreview.net/forum?id=wqD6TfbYkrn)
- code: [https://github.com/ZhaoyangLyu/Point_Diffusion_Refinement](https://github.com/ZhaoyangLyu/Point_Diffusion_Refinement)

## Presentations

**[1] Geometry-Consistent Neural Shape Representation with Implicit Displacement Fields**
- intro: ICLR 2022, compared with SIREN, high and low sepctral
- paper: https://openreview.net/forum?id=yhCp5RcZD7
- github: [https://github.com/yifita/idf](https://github.com/yifita/idf)

**[2] Adaptive Wavelet Transformer Network for 3D Shape Representation Learning**
- intro: ICLR2022, Wavelet
- paper: [https://openreview.net/forum?id=5MLb3cLCJY](https://openreview.net/forum?id=5MLb3cLCJY)
- no code



## Adversarial Learning
**[1] 3D Adversarial Attacks Beyond Point Cloud**
- intro: Tsinghua, 
- Compared attacks: kNN, GeoA3
- arXiv: https://arxiv.org/pdf/2104.12146.pdf
- github: [https://github.com/cuge1995/Mesh-Attack](https://github.com/cuge1995/Mesh-Attack)

**[2] PointCutMix: Regularization Strategy for Point Cloud Classification**
- intro: beijing,  a simple and effective augmentation 
- Compared attacks: kNN, GeoA3
- arXiv: https://arxiv.org/pdf/2101.01461.pdf
- github: [https://github.com/cuge1995/PointCutMix](https://github.com/cuge1995/PointCutMix)

**[3] Generating 3D Adversarial Point Clouds**
- intro: point perturbation attack, Hausdorff Distance, Chamfer Distance
- arXiv: https://arxiv.org/pdf/1809.07016.pdf
- github: [https://github.com/xiangchong1/3d-adv-pc](https://github.com/xiangchong1/3d-adv-pc)

**[4] Robust Adversarial Objects against Deep Learning Models**
- intro: kNN attacks
- arXiv: https://ojs.aaai.org//index.php/AAAI/article/view/5443
- github: [https://github.com/jinyier/ai_pointnet_attack](https://github.com/jinyier/ai_pointnet_attack)

**[5] PointCloud Saliency Maps**
- intro: point dropping attack
- arXiv: https://openaccess.thecvf.com/content_ICCV_2019/papers/Zheng_PointCloud_Saliency_Maps_ICCV_2019_paper.pdf
- github: [https://github.com/tianzheng4/PointCloud-Saliency-Maps](https://github.com/tianzheng4/PointCloud-Saliency-Maps)

**[6] IF-Defense: 3D Adversarial Point Cloud Defense via Implicit Function based Restoration**
- intro: rejected from ICLR 2022
- arXiv: https://arxiv.org/pdf/2010.05272.pdf
- github: [https://github.com/Wuziyi616/IF-Defense](https://github.com/Wuziyi616/IF-Defense)

**[7] LG-GAN: Label Guided Adversarial Network for Flexible Targeted Attack of Point Cloud-based Deep Networks**
- intro: CVPE 2020, LG-GAN
- arXiv: https://arxiv.org/abs/2011.00566
- github: [https://github.com/RyanHangZhou/tensorflow-LG-GAN](https://github.com/RyanHangZhou/tensorflow-LG-GAN)

**[8] AdvPC: Transferable Adversarial Perturbations on 3D Point Clouds**
- intro: ECCV 2020, AdvPC
- arXiv: https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123570239.pdf
- github: [https://github.com/ajhamdi/AdvPC](https://github.com/ajhamdi/AdvPC)

**[9] LPF-Defense: 3D Adversarial Defense based on Frequency Analysis**
- intro: Frequency
- arXiv: https://arxiv.org/pdf/2202.11287.pdf
- github: [https://github.com/kimianoorbakhsh/LPF-Defense](https://github.com/kimianoorbakhsh/LPF-Defense)

**[10] PointGuard: Provably Robust 3D Point Cloud Classification**
- intro: CVPR2022, theory solid
- arXiv: https://openaccess.thecvf.com/content/CVPR2021/papers/Liu_PointGuard_Provably_Robust_3D_Point_Cloud_Classification_CVPR_2021_paper.pdf
- no code

**[11] Robust Structured Declarative Classifiers for 3D Point Clouds: Defending Adversarial Attacks with Implicit Gradients**
- intro: CVPR2022, code is good, attacks are comprehensive
- arXiv: https://arxiv.org/abs/2203.15245
- github: [https://github.com/Zhang-VISLab/pytorch-LatticePointClassifier](https://github.com/Zhang-VISLab/pytorch-LatticePointClassifier)

**[12] Provable Defense Against Clustering Attacks on 3D Point Clouds**
- intro:  AAAI 2022 Workshop AdvML, theory
- arXiv: https://openreview.net/forum?id=6gEDBV8g-Q

**[13] Minimal Adversarial Examples for Deep Learning on 3D Point Clouds**
- intro: ICCV2021
- arXiv: https://arxiv.org/abs/2203.15245
- github: [https://github.com/hkust-vgd/minimal_adversarial_pcd](https://github.com/hkust-vgd/minimal_adversarial_pcd)


**[14] Imperceptible Transfer Attack and Defense on 3D Point Cloud Classification**
- intro: TPAMI2022
- arXiv: https://arxiv.org/pdf/2111.10990.pdf
- no code

**[15] Local Aggressive Adversarial Attacks on 3D Point Cloud**
- intro: ACML2021
- arXiv: https://proceedings.mlr.press/v157/sun21a/sun21a.pdf
- github: [https://github.com/Chenfeng1271/L3A](https://github.com/Chenfeng1271/L3A)

**[16] DUP-Net: Denoiser and Upsampler Network for 3D Adversarial Point Clouds Defense**
- intro: ICCV 2019, DUP-Net
- paper: https://openaccess.thecvf.com/content_ICCV_2019/html/Zhou_DUP-Net_Denoiser_and_Upsampler_Network_for_3D_Adversarial_Point_Clouds_ICCV_2019_paper.html
- github: [https://github.com/RyanHangZhou/tensorflow-DUP-Net](https://github.com/RyanHangZhou/tensorflow-DUP-Net)




##Others

**[1] Hindering Adversarial Attacks with Implicit Neural Representations**
- intro: ICML 2022, 
- paper: https://proceedings.mlr.press/v162/rusu22a.html
- poster: https://icml.cc/media/icml-2022/Slides/18020.pdf
- github: [https://github.com/deepmind/linac](https://github.com/deepmind/linac)


**[2] TPU-GAN: Learning temporal coherence from dynamic point cloud sequences**
- intro: ICLR2022, umsampling to build up 4D features
- paper: [https://openreview.net/pdf?id=FEBFJ98FKx](https://openreview.net/pdf?id=FEBFJ98FKx)
- no code
- More about 4D point cloud: https://github.com/hehefan/Awesome-Dynamic-Point-Cloud-Analysis



## DATASET
 Manifold40, ModelNet40, ShapeNet, ScanObjectNN


| Updated: 2022/10/27|
| :---------: |

