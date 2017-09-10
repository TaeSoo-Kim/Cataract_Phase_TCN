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
ResTCN_rms_L1 | 0 | 30 | x | x | x | x | 6 | 0.0 | rms | 11721946 |  |
ResTCN_rms_L1 | 0 | 30 | x | x | x | x | 6 | 0.3 | rms | 11721893 |  |
ResTCN_rms_L1 | 0 | 30 | x | x | x | x | 6 | 0.5 | rms | 11721960 |  |
ResTCN_rms_L2 | 0 | 30 | x | x | x | x | 6 | 0.0 | rms | 11721972 |  |
ResTCN_rms_L2 | 0 | 30 | x | x | x | x | 6 | 0.3 | rms | 11721993 |  |
ResTCN_rms_L2 | 0 | 30 | x | x | x | x | 6 | 0.5 | rms | 11722007 |  |

