"""
code snippets (gists) to run in python intepreter

simplification of visualization_gist.py

intended to generate subject and population visualizations from SLF_L 
cluster model studys published to aws s3.

used for OHBM abstract.
"""

from visualizations import *
from os import makedirs
from os.path import join

bundle_name = 'SLF_L'
base_dir = join('subbundles', 'HCP_test_retest', bundle_name)
for subject in subjects:
    for session in session_names:
        makedirs(join(base_dir, subject, session), exist_ok=True)

fa_scalar_data = load_fa_scalar_data(base_dir)
md_scalar_data = load_md_scalar_data(base_dir)
tractograms = load_tractograms(base_dir, bundle_name)
model_names, _, cluster_idxs, cluster_names, _ = load_clusters(base_dir, bundle_name)

############
# profile plots

import pickle
from os.path import exists

f_name = 'cluster_afq_fa_profiles.pkl'

if exists(join(base_dir, f_name)):
    with open(join(base_dir, f_name), 'rb') as handle:
        cluster_afq_fa_profiles = pickle.load(handle)
else:
    cluster_afq_fa_profiles = get_cluster_afq_profiles(fa_scalar_data, cluster_names, cluster_idxs, tractograms)
    with open(join(base_dir, f_name), 'wb') as handle:
        pickle.dump(cluster_afq_fa_profiles, handle)

# plot_cluster_reliability(base_dir, bundle_name, 'fa', cluster_afq_fa_profiles, model_names, cluster_names)

f_name = 'cluster_afq_md_profiles.pkl'

if exists(join(base_dir, f_name)):
    with open(join(base_dir, f_name), 'rb') as handle:
        cluster_afq_md_profiles = pickle.load(handle)
else:
    cluster_afq_md_profiles = get_cluster_afq_profiles(md_scalar_data, cluster_names, cluster_idxs, tractograms)
    with open(join(base_dir, f_name), 'wb') as handle:
        pickle.dump(cluster_afq_md_profiles, handle)

# plot_cluster_reliability(base_dir, bundle_name, 'md', cluster_afq_md_profiles, model_names, cluster_names)

############
# reliability plot

import pickle
from os.path import exists

f_name = 'bundle_dice_coef.pkl'
if exists(join(base_dir, f_name)):
    with open(join(base_dir, f_name), 'rb') as handle:
        bundle_dice_coef = pickle.load(handle)
else:
    bundle_dice_coef = get_bundle_dice_coefficients(base_dir, tractograms)
    with open(join(base_dir, f_name), 'wb') as handle:
        pickle.dump(bundle_dice_coef, handle)

f_name = 'cluster_dice_coef.pkl'
if exists(join(base_dir, f_name)):
    with open(join(base_dir, f_name), 'rb') as handle:
        cluster_dice_coef = pickle.load(handle)
else:
    cluster_dice_coef = get_cluster_dice_coefficients(base_dir, bundle_name, model_names, cluster_names)
    with open(join(base_dir, f_name), 'wb') as handle:
        pickle.dump(cluster_dice_coef, handle)

f_name = 'bundle_profile_fa_r2.pkl'
if exists(join(base_dir, f_name)):
    with open(join(base_dir, f_name), 'rb') as handle:
        bundle_profile_fa_r2 = pickle.load(handle)
else:
    bundle_profile_fa_r2 = get_bundle_reliability(base_dir, 'fa', fa_scalar_data, tractograms)
    with open(join(base_dir, f_name), 'wb') as handle:
        pickle.dump(bundle_profile_fa_r2, handle)

f_name = 'cluster_profile_fa_r2.pkl'
if exists(join(base_dir, f_name)):
    with open(join(base_dir, f_name), 'rb') as handle:
        cluster_profile_fa_r2 = pickle.load(handle)
else:
    cluster_profile_fa_r2 = get_cluster_reliability(base_dir, bundle_name, cluster_afq_fa_profiles, cluster_names)
    with open(join(base_dir, f_name), 'wb') as handle:
        pickle.dump(cluster_profile_fa_r2, handle)

f_name = 'bundle_profile_md_r2.pkl'
if exists(join(base_dir, f_name)):
    with open(join(base_dir, f_name), 'rb') as handle:
        bundle_profile_md_r2 = pickle.load(handle)
else:
    bundle_profile_md_r2 = get_bundle_reliability(base_dir, 'md', md_scalar_data, tractograms)
    with open(join(base_dir, f_name), 'wb') as handle:
        pickle.dump(bundle_profile_md_r2, handle)

f_name = 'cluster_profile_md_r2.pkl'
if exists(join(base_dir, f_name)):
    with open(join(base_dir, f_name), 'rb') as handle:
        cluster_profile_md_r2 = pickle.load(handle)
else:
    cluster_profile_md_r2 = get_cluster_reliability(base_dir, bundle_name, cluster_afq_md_profiles, cluster_names)
    with open(join(base_dir, f_name), 'wb') as handle:
        pickle.dump(cluster_profile_md_r2, handle)

population_visualizations(base_dir, bundle_name, bundle_dice_coef, cluster_dice_coef, bundle_profile_fa_r2, cluster_profile_fa_r2, bundle_profile_md_r2, cluster_profile_md_r2, model_names, cluster_names)

