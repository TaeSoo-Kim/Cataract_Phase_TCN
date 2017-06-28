
# Cataract_Phase_Image_ResNet50
- This is the balanced data version.
- ResNet50 is finetuned from imagenet pretrain model.
- 500 frames per class (task) per video. There are 11 classes (class 0: background). So, at most 5500 frames per video, 50 videos in total.
- Leave one out train test split.

### Wilmer Data image phase: 
Model | Augment  | Training Loss | Testing Loss | Validation Acc |  Depth | Dropout | Opti | SLURM ID| Notes |
---|:---:|:---:|:---:|:---:|:---:|:---:|:---: | :---:  |:---: |
ResNet50_pretrain | 0 | x | x | x | x | 50 | x | SGD | 4299209 | test=051 |
