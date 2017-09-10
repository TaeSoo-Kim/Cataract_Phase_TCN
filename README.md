# Cataract_Phase_TCN

### Wilmer Data image phase: convlen=8, sample_weighted=yes, class_weighted=yes
Model | Augment | skip rate | Training Loss | Testing Loss | Validation Acc |  Depth | Dropout | Opti | SLURM ID| Notes |
---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---: | :---:  |:---: |
Vanilla TCN | 0 | 30 | 0.918 | 0.775 | 0.163 | 6 | 0.3 | SGD | 4299209 | test=051 |
Vanilla TCN + RMS | 0 | 30 | 0.917 | 0.665 | 0.787 | 6 | 0.3 | RMS | 4299206 | test=051 | 
Vanilla TCN + ADAM | 0 | 30 | 0.751 | 0.754 | 0.878 | 6 | 0.3 | ADAM | 4299214 | test=051 | 




### Wilmer Data True length temporal phase: convlen=8
Model | Augment | skip rate | Training Loss | Testing Loss | Validation Acc |  Depth | Dropout | Opti | SLURM ID| Notes |
---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---: | :---:  |:---: |
ResTCN_rms_L1 | 0 | 30 | 0.3464 | 0.9696 | 2.0273 | 0.6728 | 6 | 0.0 | rms | 11721946 |  |
ResTCN_rms_L1 | 0 | 30 | 0.4999 | 0.9377 | 2.4585 | 0.6849 | 6 | 0.3 | rms | 11721893 |  |
ResTCN_rms_L1 | 0 | 30 | 0.5912 | 0.9294 | 2.4802 | 0.6781 | 6 | 0.5 | rms | 11721960 |  |
ResTCN_rms_L2 | 0 | 30 | 0.2969 | 0.9430 | 2.0286 | 0.6845 | 6 | 0.0 | rms | 11721972 |  |
ResTCN_rms_L2 | 0 | 30 | 0.2753 | 0.9562 | 2.0530 | **0.7402** | 6 | 0.3 | rms | 11721993 |  |
ResTCN_rms_L2 | 0 | 30 | 0.0952 | 0.9816 | 2.5015 | 0.6823 | 6 | 0.5 | rms | 11722007 |  |

