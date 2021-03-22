"""
code snippets (gists) to run in python intepreter

code to ensure was correctly calculating cluster profile confidence intervals
"""

### rerun profile plots
from visualizations import *
from os import makedirs
from os.path import join
bundle_name = 'SLF_L'
base_dir = join('subbundles', 'HCP_test_retest_MASE_DTI', bundle_name)

# get data
fa_scalar_data = load_fa_scalar_data(base_dir)
tractograms = load_tractograms(base_dir, bundle_name)
model_names, _, cluster_idxs, cluster_names, _ = load_clusters(base_dir, bundle_name)

cluster_afq_profiles = get_cluster_afq_profiles(fa_scalar_data, cluster_names, cluster_idxs, tractograms)

from os.path import exists, join
import numpy as np
import pandas as pd
from dipy.stats.analysis import afq_profile, gaussian_weights
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.DataFrame(columns=["session", "model_name", "cluster_name", "subject", "profile"])

for subject in subjects:
    for session in session_names:
        ii = 0
        for model_name, model_cluster_names in zip(model_names[subject][session], cluster_names[subject][session]):
            for cluster_name in model_cluster_names:
                profile = cluster_afq_profiles[subject][session][ii]
                df = df.append({
                    'session': session,
                    'model_name': model_name,
                    'cluster_name': cluster_name,
                    'subject': subject,
                    'profile': profile
                }, ignore_index=True)
                ii += 1

# profile plot: mdf + fa r2
for session in session_names:
    session_model_names = df.model_name.unique()
    model_name = 'mase_fa_r2_is_mdf'
    model_cluster_names = df.query(f'session == "{session}" & model_name == "{model_name}"')['cluster_name'].unique()
    plt.figure()
    for cluster_name in model_cluster_names:
        df1 = pd.DataFrame(np.array([profile for _, profile in df.query(f'session == "{session}" & model_name == "{model_name}" & cluster_name == {cluster_name}')['profile'].iteritems()]))
        df2 = pd.melt(frame=df1, var_name='node', value_name='fa_value')
        sns.lineplot(data=df2, x='node', y='fa_value')
    plt.rc('axes', labelsize=14)
    plt.show()
    # plt.savefig(join(base_dir, f'{session}_{model_name}_cluster_profile_ci_no_title.png'))
    # plt.close()

# different filters on dataframe
# df1 = pd.DataFrame(np.array([profile for _, profile in df.query("cluster_name == 0")["profile"].iteritems()]))
# df1 = pd.DataFrame(np.array([profile for _, profile in df.query("model_name == 'mase_fa_r2_is_mdf' & cluster_name == 0")["profile"].iteritems()]))
# df1 = pd.DataFrame(np.array([profile for _, profile in df.query("session == 'HCP_1200' & model_name == 'mase_fa_r2_is_mdf' & cluster_name == 0")["profile"].iteritems()]))

# plot all cluster profiles
df0 = pd.DataFrame(np.array([profile for _, profile in df.query("session == 'HCP_1200' & model_name == 'mase_fa_r2_is_mdf' & cluster_name == 0")["profile"].iteritems()]))
df1 = pd.DataFrame(np.array([profile for _, profile in df.query("session == 'HCP_1200' & model_name == 'mase_fa_r2_is_mdf' & cluster_name == 1")["profile"].iteritems()]))
df2 = pd.DataFrame(np.array([profile for _, profile in df.query("session == 'HCP_1200' & model_name == 'mase_fa_r2_is_mdf' & cluster_name == 2")["profile"].iteritems()]))

sns.lineplot(data=df0.T, alpha=0.05, palette={'tab:blue'}, legend=False, dashes=False)
sns.lineplot(data=df0.mean().T, color='tab:blue', legend=False)
plt.show()

sns.lineplot(data=df1.T, alpha=0.05, palette={'tab:orange'}, legend=False, dashes=False)
sns.lineplot(data=df1.mean().T, color='tab:orange', legend=False)
plt.show()

sns.lineplot(data=df2.T, alpha=0.05, palette={'tab:green'}, legend=False, dashes=False)
sns.lineplot(data=df2.mean().T, color='tab:green', legend=False)
plt.show()

# all in one
sns.lineplot(data=df0.T, alpha=0.05, palette={'tab:blue'}, legend=False, dashes=False)
sns.lineplot(data=df0.mean().T, color='tab:blue', legend=False)
sns.lineplot(data=df1.T, alpha=0.05, palette={'tab:orange'}, legend=False, dashes=False)
sns.lineplot(data=df1.mean().T, color='tab:orange', legend=False)
sns.lineplot(data=df2.T, alpha=0.05, palette={'tab:green'}, legend=False, dashes=False)
sns.lineplot(data=df2.mean().T, color='tab:green', legend=False)
plt.show()


# bundle mean and std
import numpy as np
bundle_profile_fa_r2 = get_bundle_reliability(base_dir, fa_scalar_data, tractograms)
np.mean(list(bundle_profile_fa_r2.values()))
np.std(list(bundle_profile_fa_r2.values()))

# calculating mean, std, and CI
# pd.melt(frame=df1, var_name='node', value_name='fa_value')

df0.describe().T
df0.stack().describe()

# truncate ends
# df1.values[:,10:-10]

# mean
# df1.values.mean()

# confidence intervals
def ci_wp(a):
    """calculate confidence interval using Wikipedia's formula"""
    m = np.mean(a)
    s = 1.96*np.std(a)/np.sqrt(len(a))
    return m - s, m + s

df0 = pd.DataFrame(np.array([profile for _, profile in df.query("model_name == 'mase_fa_r2_is_mdf' & cluster_name == 0")["profile"].iteritems()]))
df1 = pd.DataFrame(np.array([profile for _, profile in df.query("model_name == 'mase_fa_r2_is_mdf' & cluster_name == 1")["profile"].iteritems()]))
df2 = pd.DataFrame(np.array([profile for _, profile in df.query("model_name == 'mase_fa_r2_is_mdf' & cluster_name == 2")["profile"].iteritems()]))

float("{0:.2f}".format(df0.values.mean()))
# sns.utils.ci(df0.values)
# sns.utils.ci(sns.algorithms.bootstrap(df0.values))
[float("{0:.2f}".format(n)) for n in ci_wp(df0.values)]
float("{0:.2f}".format(df1.values.mean()))
# sns.utils.ci(df1.values)
# sns.utils.ci(sns.algorithms.bootstrap(df21.values))
[float("{0:.2f}".format(n)) for n in ci_wp(df1.values)]
float("{0:.2f}".format(df2.values.mean()))
# sns.utils.ci(df2.values)
# sns.utils.ci(sns.algorithms.bootstrap(df2.values))
[float("{0:.2f}".format(n)) for n in ci_wp(df2.values)]

# sns.utils.ci(df1.values, axis=0)

# plt.plot(range(100), sns.utils.ci(sns.algorithms.bootstrap(df1.values), axis=0).T)
# plt.show()