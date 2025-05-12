#!/bin/bash
. /etc/profile.d/modules.sh

module load anaconda
source activate MHCex38


#### peptide embedding
# python /exports/csce/eddie/inf/groups/ajitha_project/piyush/MHCCBM/MHCCBM/components/PG/scripts/peptide_embeddings.py

#### Hyperopts
# python /exports/csce/eddie/inf/groups/ajitha_project/piyush/MHCCBM/MHCCBM/components/PG/scripts/TrainPGPredictor_CNN.py --config_path /exports/csce/eddie/inf/groups/ajitha_project/piyush/MHCCBM/MHCCBM/components/PG/config/hyperparameter_tuning/CNN/config_$1.json --save_model 0 --fold $2 --flank $3

#### Final model training
python /exports/csce/eddie/inf/groups/ajitha_project/piyush/MHCCBM/MHCCBM/components/PG/scripts/TrainPGPredictor_CNNfinalmodel.py --config_path /exports/csce/eddie/inf/groups/ajitha_project/piyush/MHCCBM/MHCCBM/components/PG/config/hyperparameter_tuning/CNN/config_$1.json --save_model 1 --flank $2