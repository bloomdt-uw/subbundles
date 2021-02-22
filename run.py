############
# process multiple bundles

from visualizations import *
from os import makedirs
from os.path import join

for bundle_name in ['SLF_L', 'SLF_R', 'ARC_L', 'ARC_R', 'CST_L', 'CST_R']:
    base_dir = join('subbundles', 'HCP_test_retest', bundle_name)
    for subject in subjects:
        for session in session_names:
            makedirs(join(base_dir, subject, session), exist_ok=True)
    fa_scalar_data = load_fa_scalar_data(base_dir, True)
    md_scalar_data = load_md_scalar_data(base_dir, True)
    tractograms = load_tractograms(base_dir, bundle_name)
    model_names, _, cluster_idxs, cluster_names, _ = load_clusters(base_dir, bundle_name)
    cluster_afq_fa_profiles = get_cluster_afq_profiles(fa_scalar_data, cluster_names, cluster_idxs, tractograms)
    plot_cluster_reliability(base_dir, bundle_name, 'fa', cluster_afq_fa_profiles, model_names, cluster_names)
    cluster_afq_md_profiles = get_cluster_afq_profiles(md_scalar_data, cluster_names, cluster_idxs, tractograms)
    plot_cluster_reliability(base_dir, bundle_name, 'md', cluster_afq_md_profiles, model_names, cluster_names)
    bundle_dice_coef = get_bundle_dice_coefficients(base_dir, tractograms)
    cluster_dice_coef = get_cluster_dice_coefficients(base_dir, bundle_name, model_names, cluster_names)
    bundle_profile_fa_r2 = get_bundle_reliability(base_dir, fa_scalar_data, tractograms)
    cluster_profile_fa_r2 = get_cluster_reliability(base_dir, bundle_name, cluster_afq_fa_profiles, cluster_names)
    bundle_profile_md_r2 = get_bundle_reliability(base_dir, md_scalar_data, tractograms)
    cluster_profile_md_r2 = get_cluster_reliability(base_dir, bundle_name, cluster_afq_md_profiles, cluster_names)
    population_visualizations(base_dir, bundle_name, bundle_dice_coef, cluster_dice_coef, bundle_profile_fa_r2, cluster_profile_fa_r2, bundle_profile_md_r2, cluster_profile_md_r2, model_names, cluster_names)

############
# anatomy

# TODO segfault 11
from visualizations import *
from os.path import join
bundle_name = 'CST_R'
base_dir = join('subbundles', 'HCP_test_retest', bundle_name)
subjects = ['125525', '175439', '562345']
model_names, _, cluster_idxs, cluster_names, _ = load_clusters(base_dir, bundle_name)
tractograms = load_tractograms(base_dir, bundle_name)
anatomy_visualizations(base_dir, bundle_name, subjects[2], model_names, cluster_names, cluster_idxs, tractograms)


############
# cluster counts
import pandas as pd
import ast
df = pd.read_csv('subbundles/HCP_test_retest/SLF_L/cluster_counts.csv')
df = df.iloc[:,1:].applymap(lambda x: ast.literal_eval(x.replace('array(', '').replace(')','')))
df1 = df[subjects]
df1 = df1.T
df1.columns = ['HCP_1200', 'HCP_Retest']
df1 = df1.T
with pd.option_context('display.max_colwidth', -1):
    df1

############
# cluster size differences between runs

import pandas as pd
import ast
import numpy as np

df1 = pd.read_csv('subbundles/HCP_test_retest_SC_MASE_NO_DTW/SLF_L/cluster_counts.csv')
df1 = df1.iloc[:,1:].applymap(lambda x: ast.literal_eval(x.replace('array(', '').replace(')',''))[10])

df2 = pd.read_csv('subbundles/HCP_test_retest/SLF_L/cluster_counts.csv')
df2 = df2.iloc[:,1:].applymap(lambda x: ast.literal_eval(x.replace('array(', '').replace(')',''))[1])

df = pd.concat([df1,df2]).T.loc[pd.concat([df1.T,df2.T]).astype(str).drop_duplicates(keep=False).index.unique()]
df.columns = ['HCP_1200 0', 'HCP_Retest 0', 'HCP_1200 1', 'HCP_Retest 1']

pd.DataFrame({'HCP_1200':df.apply(lambda x: np.subtract(x['HCP_1200 0'],x['HCP_1200 1']), axis=1), 'HCP_Retest':df.apply(lambda x: np.subtract(x['HCP_Retest 0'],x['HCP_Retest 1']), axis=1)})

############

import importlib
importlib.reload(visualizations)