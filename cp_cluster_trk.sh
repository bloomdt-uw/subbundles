#!/bin/bash
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