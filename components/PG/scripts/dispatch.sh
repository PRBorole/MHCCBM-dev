#!/bin/bash
# Initialise the environment modules
. /etc/profile.d/modules.sh


# # ##### for peptide embedding
# qsub -e /exports/csce/eddie/inf/groups/ajitha_project/piyush/MHCCBM/MHCCBM/components/PG/scripts/errout/errout_peptidembedding_flank0.err -o /exports/csce/eddie/inf/groups/ajitha_project/piyush/MHCCBM/MHCCBM/components/PG/scripts/errout/errout_peptidembedding_flank0.out -P inf_ajitha_group -l h_vmem=16G -l rl9=true /exports/csce/eddie/inf/groups/ajitha_project/piyush/MHCCBM/MHCCBM/components/PG/scripts/individual_dispatch.sh   

## for hyperopts
# for ((i = 0; i <= 107; i++)); do
#     for ((fold = 0; fold <= 4; fold++)); do
#         qsub -e /exports/csce/eddie/inf/groups/ajitha_project/piyush/MHCCBM/MHCCBM/components/PG/scripts/errout/CNN_hyperopts/errout_cv5_fold${fold}_CNN$i.err -o /exports/csce/eddie/inf/groups/ajitha_project/piyush/MHCCBM/MHCCBM/components/PG/scripts/errout/CNN_hyperopts/errout_cv5_fold${fold}_CNN$i.out -P inf_ajitha_group -l h_vmem=32G -l rl9=true -l h_rt=160:00:00  /exports/csce/eddie/inf/groups/ajitha_project/piyush/MHCCBM/MHCCBM/components/PG/scripts//individual_dispatch.sh $i $fold   
#     done
# done

########### for testing flank length
# i=40 #combination best from hyperopts
# for ((flank = 0; flank <= 5; flank++)); do
#     for ((fold = 0; fold <= 4; fold++)); do
#         qsub -e /exports/csce/eddie/inf/groups/ajitha_project/piyush/MHCCBM/MHCCBM/components/PG/scripts/errout/errout_cv5_fold${fold}_CNN${i}_flank${flank}.err -o /exports/csce/eddie/inf/groups/ajitha_project/piyush/MHCCBM/MHCCBM/components/PG/scripts/errout/errout_cv5_fold${fold}_CNN${i}_flank${flank}.out -P inf_ajitha_group -pe sharedmem 8 -l h_vmem=32G -l rl9=true -l h_rt=36:00:00  /exports/csce/eddie/inf/groups/ajitha_project/piyush/MHCCBM/MHCCBM/components/PG/scripts//individual_dispatch.sh $i $fold $flank
#     done
# done

# ### for final training
i=40 #combination best from hyperopts
for ((flank = 0; flank <= 5; flank++)); do
    qsub -e /exports/csce/eddie/inf/groups/ajitha_project/piyush/MHCCBM/MHCCBM/components/PG/scripts/errout/final_model/errout_65iter_finalModel_CNN${i}_flank${flank}.err -o /exports/csce/eddie/inf/groups/ajitha_project/piyush/MHCCBM/MHCCBM/components/PG/scripts/errout/final_model/errout_65iter_finalModel_CNN${i}_flank${flank}.out -P inf_ajitha_group -pe sharedmem 8 -l h_vmem=32G -l rl9=true -l h_rt=12:00:00  /exports/csce/eddie/inf/groups/ajitha_project/piyush/MHCCBM/MHCCBM/components/PG/scripts//individual_dispatch.sh $i $flank
done
