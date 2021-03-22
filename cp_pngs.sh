#!/bin/bash
# one off script to download all pairplots, silhouette_scores, and bic_scores
# for the MASE_ChooseK_Study for every bundle and subject.
# 
# NOTE: to prevent file name collisions renames filename to include full path
# by replacing path separtator with underscore.
#
# WARNING: this script was used when only running for one bundle and a few
# individual subjects.
#
# now that functional just use the subbundle_choose_k.ipynb
for fname in `aws s3 ls --recursive s3://hcp-subbundle/MASE_ChooseK_Study/ | awk '{print $4}' | grep 'png'`
do 
    local_fname=`echo $fname | tr / _`
    aws s3 cp s3://hcp-subbundle/$fname $local_fname
done