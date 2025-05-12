#!/bin/bash
# Initialise the environment modules
. /etc/profile.d/modules.sh


### for hyperopts
for ((i = 0; i <= 121; i++)); do
    qsub -e /exports/csce/eddie/inf/groups/ajitha_project/piyush/MHCCBM/MHCCBM/components/TD/scripts/errout/CNN_hyperopts/errout_cv5_CNN$i.err -o /exports/csce/eddie/inf/groups/ajitha_project/piyush/MHCCBM/MHCCBM/components/TD/scripts/errout/CNN_hyperopts/errout_cv5_CNN$i.out -P inf_ajitha_group -l h_vmem=16G -l rl9=true /exports/csce/eddie/inf/groups/ajitha_project/piyush/MHCCBM/MHCCBM/components/TD/scripts//individual_dispatch.sh $i    
done


