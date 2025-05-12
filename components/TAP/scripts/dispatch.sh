#!/bin/bash
# Initialise the environment modules
. /etc/profile.d/modules.sh


# ##### for peptide embedding
# qsub -e /exports/csce/eddie/inf/groups/ajitha_project/piyush/MHCCBM/MHCCBM/components/TAP/scripts/errout/errout.err -o /exports/csce/eddie/inf/groups/ajitha_project/piyush/MHCCBM/MHCCBM/components/TAP/scripts/errout/errout.out -P inf_ajitha_group -l h_vmem=8G -l rl9=true /exports/csce/eddie/inf/groups/ajitha_project/piyush/MHCCBM/MHCCBM/components/TAP/scripts//individual_dispatch.sh   

### for hyperopts
for ((i = 0; i <= 97; i++)); do
    qsub -e /exports/csce/eddie/inf/groups/ajitha_project/piyush/MHCCBM/MHCCBM/components/TAP/scripts/errout/CNN_hyperopts/errout_cv5_CNN$i.err -o /exports/csce/eddie/inf/groups/ajitha_project/piyush/MHCCBM/MHCCBM/components/TAP/scripts/errout/CNN_hyperopts/errout_cv5_CNN$i.out -P inf_ajitha_group -l h_vmem=16G -l rl9=true /exports/csce/eddie/inf/groups/ajitha_project/piyush/MHCCBM/MHCCBM/components/TAP/scripts//individual_dispatch.sh $i    
done


