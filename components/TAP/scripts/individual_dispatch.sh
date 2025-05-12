#!/bin/bash
. /etc/profile.d/modules.sh

module load anaconda
source activate MHCex38

# python /exports/csce/eddie/inf/groups/ajitha_project/piyush/MHCCBM/MHCCBM/components/TAP/scripts/peptide_embeddings.py


python /exports/csce/eddie/inf/groups/ajitha_project/piyush/MHCCBM/MHCCBM/components/TAP/scripts/TrainTAPPredictor_CNN.py --config_path /exports/csce/eddie/inf/groups/ajitha_project/piyush/MHCCBM/MHCCBM/components/TAP/config/hyperparameter_tuning/CNN/config_$1.json --save_model 0

# python /exports/csce/eddie/inf/groups/ajitha_project/piyush/MHCCBM/MHCCBM/components/TAP/scripts/TrainTAPPredictor_CNN.py --config_path /exports/csce/eddie/inf/groups/ajitha_project/piyush/MHCCBM/MHCCBM/components/TAP/config/final_configs_5runs/run$1.json --save_model 1