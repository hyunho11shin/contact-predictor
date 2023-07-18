# contact-predictor

CSSB 2023 Summer Study

### Hello IU
```
#!/bin/bash
#SBATCH -p gpu
#SBATCH --mem=32g
#SBATCH --gres=gpu:A5000:2
#SBATCH -c 4
#SBATCH --output=helloiu.out

sleep 10s
echo "Hello IU"
```
Alternatively,
```
qlogin -p gpu -c 4 --mem 32g --gres=gpu:A6000:2
```
