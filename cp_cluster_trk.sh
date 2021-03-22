#!/bin/bash
# one off shell script to copy test and retest tracking files from aws
# finds subjects from existing local directory
#
# WARNING aws bucket has been restructured 
#  * must specify the study in the remote path
#
# WARNING script assumptions include:
#  * local study directory exists
#  * script is run from the bundle directory in local study directory
#  * only for SLF_L bundle (could fix this with shell argument)
#  * only for number of clusters is 3
#  * only for mase fa r2 is mdf model
#
# THERE ARE NO CHECKS TO VALIDATE THESE ASSUMPTIONS
for d in $(ls -d */)
do 
    d=${d%?}
    echo $d
    cd $d/HCP_1200
    aws s3 cp s3://hcp-subbundle/HCP_1200/SLF_L/$d/3/mase_fa_r2_is_mdf_cluster_0.trk .
    aws s3 cp s3://hcp-subbundle/HCP_1200/SLF_L/$d/3/mase_fa_r2_is_mdf_cluster_1.trk .
    aws s3 cp s3://hcp-subbundle/HCP_1200/SLF_L/$d/3/mase_fa_r2_is_mdf_cluster_2.trk .
    cd ../HCP_Retest/
    aws s3 cp s3://hcp-subbundle/HCP_Retest/SLF_L/$d/3/mase_fa_r2_is_mdf_cluster_0.trk .
    aws s3 cp s3://hcp-subbundle/HCP_Retest/SLF_L/$d/3/mase_fa_r2_is_mdf_cluster_1.trk .
    aws s3 cp s3://hcp-subbundle/HCP_Retest/SLF_L/$d/3/mase_fa_r2_is_mdf_cluster_2.trk .
    cd ../../
done