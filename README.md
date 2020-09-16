# Geometry-Supervised-Pose-Network-for-Accurate-Retail-Shelf-Pose-Estimation
Monocular shelf posture estimation algorithm in the retail field

[[Paper]](https://ieeexplore.ieee.org/document/9112652) [[Blog]](https://www.zhihu.com/people/kris-allen-65/posts)

## Overview
Code release for Geometry Supervised Pose Network for Accurate Retail Shelf Pose Estimation.
This is a GSPN which contains a GSL Module. Using VGG as backbone, I add GSL module into vgg as a monitor to supervise the shelf pose estimatiom, a complete pose of shelf containing three value in 3D space.

This paper is jointly complished by [Yongqiang Mou](https://github.com/AIKnowU) *，[Zhiyi Huang](https://github.com/Huang9495) (JXSTNUAA / AIBC)，Lingfan Lin ， Yishi Guo and [Zhen Yang](https://github.com/yangzhen5771) (JXSTNUAA / AIBC) in IEEE Transactions on Industrial Informatics (IEEE TII, IF=9.112) 2020。

![](file:///Users/wangyabei/Pictures/img/dataset.jpg)
Further information please contact [Yongqiang Mou](yongqiang.mou@gmail.com)

## Requirements
[PyTorch](https://pytorch.org/) (version >= 1.1.0)
[Torchvision](https://pytorch.org/) (version >= 0.3.0)

## Getting Started (Testing)

### Testing

* Model-Site:
   ```
   链接:https://pan.baidu.com/s/1Pmi6bfbDok0VOlKjMOGDLA  
   密码:uhnz
   ```  
* Command:
   ```
   cd ~/project  
   mkdir models
   cp ./checkpoint.pth.tar ~/project/models
   download datasets
   mv ./traindata ~/project/dataset
   cd ~/project/code
   python python eval.py
   ```  
## Experimental Result
Basis | Method | Input | Pitch Model | Yaw Model | Roll Model | Average Error
-- | ---- | ---- | ---- | ---- | ---- | ----
FFPVA[1] | FFPVA[1] | RGB | 16.497◦ | 6.574◦ | 5.087◦ | 9.386◦
PoseNet[2] | BL-RGB | RGB | 8.533◦ | 2.991◦ | 1.345◦ | 4.289◦
PoseNet[2] | BL-LSD | GIM | 8.083◦ | 2.662◦ | 1.125◦ | 3.957◦
PoseNet[2] | BL-Fusion | RGB+GIM | 8.078◦ | 2.609◦ | 1.102◦ | 3.929◦ 
L2SD | L2SD | RGB | 7.958◦ | 1.543◦ | 1.119◦ | 3.540◦
GSPN | GSPN | RGB | 7.456◦ | 0.864◦ | 0.761◦ | 3.027◦

## Reference
[1]. J. Lezama, and G. Randall R. G. V. Gioi, and J. M. Morel, “Finding vanishing points via point alignments in image primal and dual domains,” IEEE Computer Vision and Pattern Recognition, pp. 509–515, Jun. 2014.  
[2].  A. Kendall, M. Grimes, and R. Cipolla, “Posenet: A convolutional network for real-time 6-dof camera relocalization,” IEEE International Conference on Computer Vision, p. 265, Dec. 2015.

## License and Citation
@article{0Geometry,
  title={Geometry Supervised Pose Network for Accurate Retail Shelf Pose Estimation},
  author={Mou, Yongqiang  and  Huang, Zhiyi  and  Lin, Lingfan  and  Guo, Yishi  and  Yang, Zhen},
  journal={IEEE Transactions on Industrial Informatics},
  volume={PP},
  number={99},
  pages={1-1},
  year={2020}
}
