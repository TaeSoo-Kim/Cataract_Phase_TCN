# Cataract_Phase_TCN

### Wilmer Data image phase: convlen=8, sample_weighted=yes, class_weighted=yes
Model | Augment | skip rate | Training Loss | Testing Loss | Validation Acc |  Depth | Dropout | Opti | SLURM ID| Notes |
---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---: | :---:  |:---: |
Vanilla TCN | 0 | 30 | 0.918 | 0.775 | 0.163 | 6 | 0.3 | SGD | 4299209 | test=051 |
Vanilla TCN + RMS | 0 | 30 | 0.917 | 0.665 | 0.787 | 6 | 0.3 | RMS | 4299206 | test=051 | 
Vanilla TCN + ADAM | 0 | 30 | 0.751 | 0.754 | 0.878 | 6 | 0.3 | ADAM | 4299214 | test=051 | 




### Wilmer Data real temporal phase: convlen=8, sample_weighted=no, class_weighted=no
Model | Augment | skip rate | Training Loss | Testing Loss | Validation Acc |  Depth | Dropout | Opti | SLURM ID| Notes |
---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---: | :---:  |:---: |
ResTCN_adam | 0 | 30 | x | x | x | x | 6 | 0.5 | adam | 7657684 |  |
ResTCN_rms | 0 | 30 | x | x | x | x | 6 | 0.5 | rms | 9657685 |  |

