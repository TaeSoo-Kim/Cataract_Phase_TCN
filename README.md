# Cataract_Phase_TCN

### Wilmer Data image phase: convlen=8, sample_weighted=yes, class_weighted=yes
Model | Augment | skip rate | Training Loss | Testing Loss | Validation Acc |  Depth | Dropout | Opti | SLURM ID| Notes |
---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---: | :---:  |:---: |
Vanilla TCN | 0 | 30 | x | x | x | 6 | 0.3 | SGD | 4299209 | test=051 |
Vanilla TCN + RMS | 0 | 30 | x | x | x | 6 | 0.3 | RMS | 4299206 | test=051 | 
Vanilla TCN + ADAM | 0 | 30 | x | x | x | 6 | 0.3 | ADAM | 4299214 | test=051 | 
