#!/bin/bash
. /etc/profile.d/modules.sh

module load anaconda
source activate MHCex38

# python /exports/csce/eddie/inf/groups/ajitha_project/piyush/MHCCBM/MHCCBM/TD/src/sequence_to_embedding.py

python /exports/csce/eddie/inf/groups/ajitha_project/piyush/MHCCBM/MHCCBM/components/TD/scripts/TrainTDPredictor_CNN.py --config_path /exports/csce/eddie/inf/groups/ajitha_project/piyush/MHCCBM/MHCCBM/components/TD/config/hyperparameter_tuning/CNN/config_$1.json --save_model 0