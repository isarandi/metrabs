# Model Downloads

Note: these models **cannot be used for commercial purposes** due to the license terms of the training data (see below).

| Model Download       | Backbone                 | Detector    | 3DPW PCK@50mm ↑  | 3DPW MPJPE ↓ | H36M S9/S11 MPJPE ↓ | 3DHP PCK@150mm ↑ | MuPoTS PCK@150mm ↑ | MuPoTS Abs-MPJPE ↓ | Avg FPS on 3DPW (batched) | Single-person FPS (batched) |
|----------------------|--------------------------|-------------|------------------|--------------|---------------------|------------------|--------------------|--------------------|---------------------------|-----------------------------|
| [metrabs_eff2l_y4](https://omnomnom.vision.rwth-aachen.de/data/metrabs/metrabs_eff2l_y4_20211019.zip) (654 MB)     | EfficientNetV2-L         | YOLOv4      | **53.3%**        | **61.9 mm**  | **41.1 mm**         | **95.7%**        | **94.7%**          | 191.7 mm           | TBA                       | TBA                         |
| [metrabs_eff2l_y4_360](https://omnomnom.vision.rwth-aachen.de/data/metrabs/metrabs_eff2l_y4_360_20211019.zip)* (653 MB)| EfficientNetV2-L         | YOLOv4      | 48.7%            | 66.1 mm      | 47.5 mm             | 95.0%            | 94.1%              | **187.2 mm**       | TBA                       | TBA                         |
| [metrabs_eff2s_y4](https://omnomnom.vision.rwth-aachen.de/data/metrabs/metrabs_eff2s_y4_20211026.zip) (305 MB)     | EfficientNetV2-S         | YOLOv4      | 50.8%        | 64.9 mm  | 47.4 mm         | 94.5%        | 94.1%          | 192.2 mm           | TBA                       | TBA                         |
| [metrabs_mob3l_y4t](https://omnomnom.vision.rwth-aachen.de/data/metrabs/metrabs_mob3l_y4t_20211019.zip) (40 MB)   | MobileNetV3-Large        | YOLOv4-tiny | 44.3%            | 74.1 mm      | 54.0 mm             | 89.5%            | 91.0%              | 221.3 mm           | **TBA**                   | **TBA**                     |

*metrabs_eff2l_y4_360 was trained with 360° rotation augmentation, resulting in an approximately rotation-invariant model that qualitatively handles extreme poses (e.g., yoga, cartwheel) more robustly.  

This evaluation was performed with the built-in 5-crop test-time augmentation. More detailed evaluation results TBA.

## Training Datasets

- [Human3.6M](http://vision.imar.ro/human3.6m)
- [MuCo-3DHP](https://vcai.mpi-inf.mpg.de/projects/SingleShotMultiPerson/)
- [CMU-Panoptic](http://domedb.perception.cs.cmu.edu/)
- [SAIL-VOS](http://sailvos.web.illinois.edu/_site/index.html)
- [SURREAL](https://www.di.ens.fr/willow/research/surreal/data/)
- [AIST-Dance++](https://google.github.io/aistplusplus_dataset/index.html)
- [COCO (2D pose)](https://cocodataset.org/)
