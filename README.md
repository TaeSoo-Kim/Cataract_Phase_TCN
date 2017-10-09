# Cataract_Phase_TCN

### Wilmer Data True length temporal phase: convlen=8, short clips only, feature=Resnet50,imgnet finetuned
Model | Augment | skip rate | Training Loss | Testing Loss | Validation Acc |  Dropout | Opti | SLURM ID| Notes |
---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---: | :---:  |
ResTCN_rms_L1 | 0 | 30 | 0.3464 | 2.0273 | 0.6728 | 0.0 | rms | 11721946 |  |
ResTCN_rms_L1 | 0 | 30 | 0.4999 | 2.4585 | 0.6849 | 0.3 | rms | 11721893 |  |
ResTCN_rms_L1 | 0 | 30 | 0.5912 | 2.4802 | 0.6781 | 0.5 | rms | 11721960 |  |
ResTCN_rms_L2 | 0 | 30 | 0.2969 | 2.0286 | 0.6845 | 0.0 | rms | 11721972 |  |
ResTCN_rms_L2 | 0 | 30 | 0.2753 | 2.0530 | **0.7402** | 0.3 | rms | 11721993 |  |
ResTCN_rms_L2 | 0 | 30 | 0.0952 | 2.5015 | 0.6823 | 0.5 | rms | 11722007 |  |
ResTCN_adam_L1 | 0 | 30 | 0.8546 | 2.7105 | 0.6587 | 0.0 | adam | 11930685 |  |
ResTCN_adam_L1 | 0 | 30 | 0.5982 | 2.3925 | 0.6894 | 0.3 | adam | 11932491 |  |
ResTCN_adam_L1 | 0 | 30 | 0.5761 | 2.6447 | 0.6590 | 0.5 | adam | 11933330 |  |
ResTCN_adam_L2 | 0 | 30 | 0.4116 | 2.0378 | 0.6783 |  0.0 | adam | 11926746 |  |
ResTCN_adam_L2 | 0 | 30 | 0.6246 | 1.9463 | 0.6652 |  0.3 | adam | 11928035 |  |
ResTCN_adam_L2 | 0 | 30 | 0.2697 | 2.2966 | 0.6539 | 0.5 | adam | 11928707 |  |

### Wilmer Data True length temporal phase: convlen=8, short clips only, feature=TKNET
Model | Augment | skip rate | Training Loss | Testing Loss | Validation Acc |  Dropout | Opti | SLURM ID| Notes |
---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---: | :---:  |
ResTCN_adam_L1 | 0 | 30 | x | x | x | 0.3 | adam | 13913990 | feat_dim=256 |
ResTCN_adam_L1 | 0 | 30 | x | x | x | 0.5 | adam | 13913981 | feat_dim=256 |
ResTCN_adam_L2 | 0 | 30 | x | x | x | 0.3 | adam | 13913985 | feat_dim=256 |
ResTCN_adam_L2 | 0 | 30 | x | x | x | 0.5 | adam | 13913987 | feat_dim=256 |
ResTCN_rms_L1 | 0 | 30 | x | x | x | 0.3 | rms | 13913995 | feat_dim=256 |
ResTCN_rms_L1 | 0 | 30 | x | x | x | 0.5 | rms | 13913996 | feat_dim=256 |
ResTCN_rms_L2 | 0 | 30 | x | x | x | 0.3 | rms | 13913997 | feat_dim=256 |
ResTCN_rms_L2 | 0 | 30 | x | x | x | 0.5 | rms | 13913999 | feat_dim=256 |

### Wilmer Data True length temporal phase: convlen=8, short clips only, feature=TKNET_concat
Model | Augment | skip rate | Training Loss | Testing Loss | Validation Acc |  Dropout | Opti | SLURM ID| Notes |
---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---: | :---:  |
ResTCN_adam_L1 | 0 | 30 | x | x | x | 0.3 | adam | 13914077 | feat_dim=448 |
ResTCN_adam_L1 | 0 | 30 | x | x | x | 0.5 | adam | 13914082 | feat_dim=448 |
ResTCN_adam_L2 | 0 | 30 | x | x | x | 0.3 | adam | 13914158 | feat_dim=448 |
ResTCN_adam_L2 | 0 | 30 | x | x | x | 0.5 | adam | 13914160 | feat_dim=448 |
ResTCN_rms_L1 | 0 | 30 | x | x | x | 0.3 | rms | 13914162 | feat_dim=448 |
ResTCN_rms_L1 | 0 | 30 | x | x | x | 0.5 | rms | 13914163 | feat_dim=448 |
ResTCN_rms_L2 | 0 | 30 | x | x | x | 0.3 | rms | 13914166 | feat_dim=448 |
ResTCN_rms_L2 | 0 | 30 | x | x | x | 0.5 | rms | 13914168 | feat_dim=448 |


### 2D phase, feature extractor
Model |  Training Loss | Testing Loss | Validation Acc |  Dropout | Opti | SLURM ID| Notes |
---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
TKNET | x | x | 0.5510 | 0.3 | sgd | local |   
TKNET_cat | x | x | 0.57 | 0.3 | sgd | local |  

# Cataract_Skill_TCN

### JIGSAWS, kinematic, classification, (E,I,N), LOSO
Model | filter | Training Loss | Testing Loss | Validation Acc |  Dropout | Opti | SLURM ID| Notes |
---|:---:|:---:|:---:|:---:|:---:|:---:|:---: | :---:  |
ResTCN_classification | 8 | 0.4743 | 0.6479 | 0.8750 | 0.3 | adam | local | split=1
ResTCN_classification | 8 | 1.1012 | 1.3079 | 0.6250 | 0.3 | adam | local | split=2
ResTCN_classification | 8 | 1.1282 | 5.6738 | 0.6250 | 0.3 | adam | local | split=3
ResTCN_classification | 8 | 0.9438 | 1.5444 | 0.7500 | 0.3 | adam | local | split=4
ResTCN_classification | 8 | 0.5153 | 0.7974 | 0.8750 | 0.3 | adam | local | split=5


### JIGSAWS, kinematic, classification, (E,I,N), LOUO
Model | filter | Training Loss | Testing Loss | Validation Acc |  Dropout | Opti | SLURM ID| Notes |
---|:---:|:---:|:---:|:---:|:---:|:---:|:---: | :---:  |
ResTCN_classification | 8 | 0.9522 | 1.0146 | 0.8750 | 0.3 | adam | local | split=1
ResTCN_classification | 8 | 0.4416 | 0.5868 | 0.8750 | 0.3 | adam | local | split=2
ResTCN_classification | 8 | 1.1883 | 2.0241 | 0.8750 | 0.3 | adam | local | split=3
ResTCN_classification | 8 | 0.2622 | 0.5299 | 1.0000 | 0.3 | adam | local | split=4
ResTCN_classification | 8 | 1.0577 | 0.6280 | 0.8750 | 0.3 | adam | local | split=5
ResTCN_classification | 8 | 1.0970 | 2.1066 | 0.7500 | 0.3 | adam | local | split=6
ResTCN_classification | 8 | 0.6446 | 0.8367 | 0.7500 | 0.3 | adam | local | split=7
ResTCN_classification | 8 | 0.9186 | 0.9801 | 0.7500 | 0.3 | adam | local | split=8




