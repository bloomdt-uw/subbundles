# did not complete for 130518 and 287248
# does not exist for 662551
subjects = [
    '103818', '105923', '111312', '114823', '115320',
    '122317', '125525', # '130518', 
    '135528', '137128',
    '139839', '143325', '144226', '146129', '149337',
    '149741', '151526', '158035', '169343', '172332',
    '175439', '177746', '185442', '187547', '192439',
    '194140', '195041', '200109', '200614', '204521',
    '250427', # '287248', 
    '341834', '433839', '562345',
    '599671', '601127', '627549', '660951', # '662551', 
    '783462', '859671', '861456', '877168', '917255'
]
# subjects = subjects[:2]
session_names = ['HCP_1200', 'HCP_Retest']

bundle_names = ['SLF_L']

n_clusters = 3


# def relabel_clusters(labels):
#     """
#     relabel_clusters(labels)

#     During the clustering process streamlines have been assigned a label.

#     For purposes model comparison it is convienient to relabel the clusters from highest to lowest frequency.

#     The new labels are sequential with highest frequency label starting at 0

#     Parameters
#     ----------
#     labels : ndarray of ints, 1 dimension, nonnegative ints
#         The length of `labels` is equal to number of streamlines.

#     Returns
#     -------
#     out : ndarray of ints, 1 dimension, nonnegative ints

#     Examples
#     --------
#     >>> relabel_clusters(np.array([0, 1, 1, 3, 2, 1]))
#     array([3, 0, 0, 1, 2, 0])
#     """
#     import numpy as np

#     from_values = np.flip(np.argsort(np.bincount(labels))[-(np.unique(labels).size):])
#     to_values = np.arange(from_values.size)

#     d = dict(zip(from_values, to_values))

#     new_labels = np.copy(labels)

#     for k, v in d.items():
#         new_labels[labels == k] = v

#     return new_labels


def load_scalar_data(base_dir):
    """
    load_scalar_data(base_dir)

    Load FA scalar data for each subject in `subjects`.

    Scalar data is used to calculate the `afq_profiles` for bundles and for clusters.

    It is assumed that scalar data is available on S3. In this case, it will look for
    FA values from HCP reliability single shell for each `subject` and `session`.

    Parameters
    ----------
    base_dir : str
        Local working diriectory under which FA values are stored.

        It is assumed that `base_dir` folder contains following structure:
        ├── `subject`
        │   ├── `session`

        The scalar file name for each subject session is assumed to be 'FA.nii.gz'

        If the file exists it will be loaded from disk, otherwise it will be 
        downloaded from S3.

    Returns
    -------
    out : ndarray, 2 dimensional

    """
    import s3fs
    from os.path import exists, join
    import nibabel as nib

    fs = s3fs.S3FileSystem()

    scalar_basename = 'FA.nii.gz'

    scalar_data = {}

    for subject in subjects:
        scalar_data[subject] = {}
        for session in session_names:
            scalar_filename = join(base_dir, subject, session, scalar_basename)
            if not exists(scalar_filename):
                fs.get(
                    (
                        f'profile-hcp-west/hcp_reliability/single_shell/'
                        f'{session.lower()}_afq/sub-{subject}/ses-01/'
                        f'sub-{subject}_dwi_model-DTI_FA.nii.gz'
                    ),
                    scalar_filename
                )

            scalar_data[subject][session] = nib.load(scalar_filename).get_fdata()

    return scalar_data


def load_tractograms(base_dir, bundle_name):
    import s3fs
    from os.path import exists, join
    from dipy.io.streamline import load_tractogram
    import pandas as pd
    import matplotlib.pyplot as plt

    fs = s3fs.S3FileSystem()

    tractogram_basename = f'{bundle_name}.trk'

    tractograms = {}
    streamline_counts = {}

    for subject in subjects:
        tractograms[subject] = {}
        streamline_counts[subject] = {}
        for session in session_names:
            tractogram_filename = join(base_dir, subject, session, tractogram_basename)
            if not exists(tractogram_filename):
                fs.get(
                    (
                        f'profile-hcp-west/hcp_reliability/single_shell/'
                        f'{session.lower()}_afq/sub-{subject}/ses-01/'
                        f'clean_bundles/sub-{subject}_dwi_space-RASMM_model-DTI_desc-det-afq-{bundle_name}_tractography.trk'
                    ),
                    tractogram_filename
                )          
            tractogram = load_tractogram(tractogram_filename, 'same')
            streamline_counts[subject][session] = len(tractogram.streamlines)
            tractograms[subject][session] = tractogram

    pd.DataFrame(streamline_counts).to_csv(join(base_dir, 'streamline_counts.csv'))
    pd.DataFrame(streamline_counts).T[1:].plot(kind='bar')
    plt.savefig(join(base_dir, 'streamline_counts.png'))
    plt.close()

    return tractograms


def load_clusters(base_dir, bundle_name):
    import s3fs
    from os.path import exists, join, basename, splitext
    import numpy as np
    import pandas as pd

    fs = s3fs.S3FileSystem()

    # name of the model; by convention includes abbreviation for clustering algorithm and adjacencies
    model_names = {}

    # cluster name assigned to each streamline
    cluster_labels = {}

    # for each cluster name, lists the corresponding streamline indexes
    cluster_idxs = {}

    # list of cluster names
    cluster_names = {}

    # number of streamlines assigned to each cluster name
    cluster_counts = {}

    for subject in subjects:
        model_names[subject] = {}
        cluster_labels[subject] = {}
        cluster_idxs[subject] = {}
        cluster_names[subject] = {}
        cluster_counts[subject] = {}

        for session in session_names:
            model_names[subject][session] = []
            cluster_labels[subject][session] = []
            cluster_idxs[subject][session] = []
            cluster_names[subject][session] = []
            cluster_counts[subject][session] = []

            # sorted alphabetically
            remote_cluster_filenames = fs.glob(f'hcp-subbundle/{session}/{bundle_name}/{subject}/{n_clusters}/*idx.npy')
            # NOTE Be careful BIG assumption about which models were run!
            # which is not what want...
            # want sc 0 - 10 followed by mase
            # quickhack - long term should identify and separate models and hyperparameters
            my_order = [1, 3, 4, 5, 6, 7, 8, 9, 10, 2, 0]
            remote_cluster_filenames = [remote_cluster_filenames[i] for i in my_order]
            print(remote_cluster_filenames)

            for remote_cluter_filename in remote_cluster_filenames:
                # print(subject, session, remote_cluter_filename)
                cluster_basename = basename(remote_cluter_filename)
                local_cluster_filename = join(base_dir, subject, session, cluster_basename)
                if not exists(local_cluster_filename):
                    fs.get(remote_cluter_filename, local_cluster_filename)

                cluster_rootname, _ = splitext(cluster_basename)
                cluster_rootname = cluster_rootname.rsplit('_',1)[0]
                model_names[subject][session].append(cluster_rootname)

                # sorted_cluster_labels = relabel_clusters(np.load(local_cluster_filename))
                # should now be sorted by AWS
                sorted_cluster_labels = np.load(local_cluster_filename)
                cluster_labels[subject][session].append(sorted_cluster_labels)

                cluster_names[subject][session].append(np.unique(sorted_cluster_labels))
                cluster_idxs[subject][session].append(np.array([np.where(sorted_cluster_labels == i)[0] for i in np.unique(sorted_cluster_labels)]))
                cluster_counts[subject][session].append(np.bincount(sorted_cluster_labels))

    pd.DataFrame(model_names).to_csv(join(base_dir, 'model_names.csv'))
    pd.DataFrame(cluster_names).to_csv(join(base_dir, 'cluster_names.csv'))
    pd.DataFrame(cluster_counts).to_csv(join(base_dir, 'cluster_counts.csv'))

    return (model_names, cluster_labels, cluster_idxs, cluster_names, cluster_counts)


# TODO if move bundle density map to AWS do not need this method
def density_map(tractogram):
    import numpy as np
    from dipy.io.utils import create_nifti_header, get_reference_info
    import dipy.tracking.utils as dtu
    import nibabel as nib

    affine, vol_dims, voxel_sizes, voxel_order = get_reference_info(tractogram)
    tractogram_density = dtu.density_map(tractogram.streamlines, np.eye(4), vol_dims)
    # force to unsigned 8-bit
    tractogram_density = np.uint8(tractogram_density)
    nifti_header = create_nifti_header(affine, vol_dims, voxel_sizes)
    density_map_img = nib.Nifti1Image(tractogram_density, affine, nifti_header)

    return density_map_img

def get_bundle_dice_coefficients(base_dir, tractograms):
    import time
    from os.path import exists, join
    from dipy.io.stateful_tractogram import StatefulTractogram
    import numpy as np
    import pandas as pd
    # from AFQ.utils.volume import density_map, dice_coeff
    from AFQ.utils.volume import dice_coeff
    import matplotlib.pyplot as plt

    tic = time.perf_counter()

    bundle_dice_coef = {}
    
    for subject in subjects:
        
        test_sft = StatefulTractogram.from_sft(tractograms[subject][session_names[0]].streamlines, tractograms[subject][session_names[0]])                        
        test_sft.to_vox()
        test_sft_map = density_map(test_sft)

        retest_sft = StatefulTractogram.from_sft(tractograms[subject][session_names[1]].streamlines, tractograms[subject][session_names[1]])                        
        retest_sft.to_vox()
        retest_sft_map = density_map(retest_sft)

        bundle_dice_coef[subject] = dice_coeff(test_sft_map, retest_sft_map)

    # TODO average bundle dice coefficient across subjects

    pd.DataFrame(bundle_dice_coef, index=[0]).to_csv(join(base_dir, 'bundle_dice_coef.csv'))
    pd.DataFrame(bundle_dice_coef, index=[0]).T[1:].plot(kind='bar')
    plt.savefig(join(base_dir, 'bundle_dice_coef.png'))
    plt.close()

    toc = time.perf_counter()
    print(f'total bundle dice coefficients {toc - tic:0.4f} seconds')

    return bundle_dice_coef


def get_cluster_dice_coefficients(base_dir, bundle_name, model_names, cluster_names):
    import time
    import itertools
    import s3fs
    from os.path import exists, join, basename, splitext
    import numpy as np
    import pandas as pd
    from dipy.io.stateful_tractogram import StatefulTractogram
    import nibabel as nib
    from AFQ.utils.volume import dice_coeff
    import matplotlib.pyplot as plt

    tic = time.perf_counter()

    fs = s3fs.S3FileSystem()

    cluster_dice_coef = {}

    for subject in subjects:
        # _tic = time.perf_counter()
        num_test_clusters = number_of_total_clusters(cluster_names, subject, session_names[0])
        num_retest_clusters = number_of_total_clusters(cluster_names, subject, session_names[1])
        # print(num_test_clusters, num_retest_clusters)

        dice_coef_matrix = np.zeros((num_test_clusters, num_retest_clusters))

        ii = 0
        jj = 0

        # for test_model_name, test_model_cluster_name in itertools.product(model_names[subject][session_names[0]], cluster_names[subject][session_names[0]]):
        for test_model_name, test_model_cluster_name in zip(model_names[subject][session_names[0]], cluster_names[subject][session_names[0]]):
            for test_cluster_name in test_model_cluster_name:
                test_density_map_basename = f'{test_model_name}_cluster_{test_cluster_name}_density_map.nii.gz'
                local_test_cluster_density_map_filename = join(base_dir, subject, session_names[0], test_density_map_basename)
                if not exists(local_test_cluster_density_map_filename):
                    remote_test_cluster_density_map_filename = f'hcp-subbundle/{session_names[0]}/{bundle_name}/{subject}/{n_clusters}/{test_density_map_basename}'
                    fs.get(remote_test_cluster_density_map_filename, local_test_cluster_density_map_filename)
                test_cluster_density_map = nib.load(local_test_cluster_density_map_filename)
            
                # for retest_model_name, retest_model_cluster_name in itertools.product(model_names[subject][session_names[1]], cluster_names[subject][session_names[1]]):
                for retest_model_name, retest_model_cluster_name in zip(model_names[subject][session_names[1]], cluster_names[subject][session_names[1]]):
                    for retest_cluster_name in retest_model_cluster_name:
                        retest_density_map_basename = f'{retest_model_name}_cluster_{retest_cluster_name}_density_map.nii.gz'
                        local_retest_cluster_density_map_filename = join(base_dir, subject, session_names[1], retest_density_map_basename)
                        if not exists(local_retest_cluster_density_map_filename):
                            remote_retest_cluster_density_map_filename = f'hcp-subbundle/{session_names[1]}/{bundle_name}/{subject}/{n_clusters}/{retest_density_map_basename}'
                            fs.get(remote_retest_cluster_density_map_filename, local_retest_cluster_density_map_filename)
                        retest_cluster_density_map = nib.load(local_retest_cluster_density_map_filename)
                        
                        dice_coef_matrix[ii][jj] = dice_coeff(test_cluster_density_map, retest_cluster_density_map)
                        jj += 1
                ii += 1
                jj = 0

        # _toc = time.perf_counter()
        # print(subject, f'cluster dice coefficients {_toc - _tic:0.4f} seconds')

        cluster_dice_coef[subject] = dice_coef_matrix
    
    toc = time.perf_counter()
    print(f'total cluster dice coefficients {toc - tic:0.4f} seconds')

    for subject in subjects:
        dice_coef_matrix = cluster_dice_coef[subject]

        plt.figure()
        plt.title(f'test-retest {subject} {bundle_name} dice')
        plt.imshow(dice_coef_matrix, cmap='hot', interpolation='nearest', vmin=0., vmax=1.)
        plt.colorbar()
        plt.ylabel('test')
        plt.xlabel('retest')
        plt.savefig(join(base_dir, f'{subject}_{bundle_name}_dice.png'))
        plt.close()

        pd.DataFrame(dice_coef_matrix).to_csv(join(base_dir, f'{subject}_{bundle_name}_dice.csv'))

    return cluster_dice_coef


def get_bundle_reliability(base_dir, scalar_data, tractograms):
    import time
    from os.path import exists, join
    import numpy as np
    import pandas as pd
    from dipy.stats.analysis import afq_profile, gaussian_weights
    from sklearn.metrics import r2_score
    import matplotlib.pyplot as plt

    test_retest_bundle_profile_fa_r2 = {}
    df = pd.DataFrame(columns=["subject", "time"])

    tic = time.perf_counter()
    for subject in subjects:
        _tic = time.perf_counter()
        test_fa = afq_profile(
            scalar_data[subject][session_names[0]],
            tractograms[subject][session_names[0]].streamlines,
            tractograms[subject][session_names[0]].affine,
            weights=gaussian_weights(tractograms[subject][session_names[0]].streamlines)
        )

        retest_fa = afq_profile(
            scalar_data[subject][session_names[1]],
            tractograms[subject][session_names[1]].streamlines,
            tractograms[subject][session_names[1]].affine,
            weights=gaussian_weights(tractograms[subject][session_names[1]].streamlines)
        )

        test_retest_bundle_profile_fa_r2[subject] = r2_score(test_fa, retest_fa)
        _toc = time.perf_counter()
        print(subject, f'bundle reliability {_toc - _tic:0.4f} seconds')
        df = df.append({
            'subject': subject,
            'time': f'{_toc - _tic:0.4f}'
        }, ignore_index=True)

    toc = time.perf_counter()
    print(f'bundle reliability {toc - tic:0.4f} seconds')
    df = df.append({
        'subject': 'all',
        'time': f'{toc - tic:0.4f}'
    }, ignore_index=True)
    df.to_csv(join(base_dir, 'get_bundle_reliability_time.csv'))

    pd.DataFrame(test_retest_bundle_profile_fa_r2, index=[0]).to_csv(join(base_dir, 'test_retest_bundle_profile_fa_r2.csv'))
    pd.DataFrame(test_retest_bundle_profile_fa_r2, index=[0]).T[1:].plot(kind='bar', color='tab:orange')
    plt.savefig(join(base_dir, 'test_retest_bundle_profile_fa_r2.png'))
    plt.close()

    return test_retest_bundle_profile_fa_r2

def get_cluster_afq_profiles(scalar_data, cluster_names, cluster_idxs, tractograms):
    from dipy.stats.analysis import afq_profile, gaussian_weights
    import time

    tic = time.perf_counter()

    cluster_afq_profiles = {}
    for subject in subjects:
        cluster_afq_profiles[subject] = {}
        for session in session_names:
            cluster_afq_profiles[subject][session] = {}
            ii = 0
            for model_cluster_name, model_cluster_idxs in zip(cluster_names[subject][session], cluster_idxs[subject][session]):
                for cluster_name in model_cluster_name:
                    fa_scalar_data = scalar_data[subject][session]
                    cluster_streamlines = tractograms[subject][session].streamlines[model_cluster_idxs[cluster_name]]
                    cluster_affine = tractograms[subject][session].affine

                    cluster_profile = afq_profile(
                        fa_scalar_data,
                        cluster_streamlines,
                        cluster_affine,
                        weights=gaussian_weights(cluster_streamlines)
                    )

                    cluster_afq_profiles[subject][session][ii] = cluster_profile
                    ii += 1
    
    toc = time.perf_counter()
    print(f'cluster afq_profiles {toc - tic:0.4f} seconds')

    return cluster_afq_profiles

def get_cluster_reliability(base_dir, bundle_name, cluster_afq_profiles, cluster_names):
    import time
    from os.path import exists, join
    import numpy as np
    import pandas as pd
    from dipy.stats.analysis import afq_profile, gaussian_weights
    import matplotlib.pyplot as plt

    cluster_profiles = {}
    df = pd.DataFrame(columns=["subject", "time"])
    tic = time.perf_counter()

    for subject in subjects:
        _tic = time.perf_counter()
        cluster_profiles[subject] = {}

        num_test_clusters = number_of_total_clusters(cluster_names, subject, session_names[0])
        num_retest_clusters = number_of_total_clusters(cluster_names, subject, session_names[1])

        profile_matrix = np.zeros((num_test_clusters, num_retest_clusters))

        ii = 0
        jj = 0

        for test_model_cluster_name in cluster_names[subject][session_names[0]]:
            for test_cluster_name in test_model_cluster_name:
                test_cluster_profile = cluster_afq_profiles[subject][session_names[0]][ii]

                for retest_model_cluster_name in cluster_names[subject][session_names[1]]:
                    for retest_cluster_name in retest_model_cluster_name:
                        retest_cluster_profile = cluster_afq_profiles[subject][session_names[1]][jj]

                        test_retest_corr_matrix = pd.DataFrame(zip(*[test_cluster_profile, retest_cluster_profile]), columns=session_names).corr()

                        # select only the upper triangle off diagonals of the correlation matrix
                        test_retest_corr = test_retest_corr_matrix.where(np.triu(np.ones(test_retest_corr_matrix.shape), 1).astype(np.bool)).stack()

                        profile_matrix[ii][jj] = test_retest_corr
                        jj += 1
                ii += 1
                jj = 0

        cluster_profiles[subject] = profile_matrix
        _toc = time.perf_counter()
        print(subject, f'cluster reliability {_toc - _tic:0.4f} seconds')
        df = df.append({
            'subject': subject,
            'time': f'{_toc - _tic:0.4f}'
        }, ignore_index=True)

        plt.figure()
        plt.title(f'test-retest {subject} {bundle_name} profile correlation')
        plt.imshow(profile_matrix, cmap='hot', interpolation='nearest', vmin=0., vmax=1.)
        plt.colorbar()
        plt.ylabel('test')
        plt.xlabel('retest')
        plt.savefig(join(base_dir, f'{subject}_{bundle_name}_profile_corr.png'))
        plt.close()

        pd.DataFrame(profile_matrix).to_csv(join(base_dir, f'{subject}_{bundle_name}_profile_corr.csv'))

    toc = time.perf_counter()
    print(f'cluster reliability {toc - tic:0.4f} seconds')
    df = df.append({
        'subject': 'all',
        'time': f'{toc - tic:0.4f}'
    }, ignore_index=True)
    df.to_csv(join(base_dir, 'get_cluster_reliability_time.csv'))

    return cluster_profiles


def plot_cluster_reliability(base_dir, bundle_name, cluster_afq_profiles, model_names, cluster_names):
    '''plot of test and retest mean afq profiles and 25-75 confidence intervals per cluster in each model'''
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
    
    # TODO streamline cluster plots
    # cluster_colors = sns.color_palette("bright6")

    # for session in session_names:
    #     session_model_names = df.model_name.unique()
    #     for model_name in session_model_names:
    #         model_cluster_names = df.query(f'session == "{session}" & model_name == "{model_name}"')['cluster_name'].unique()

    #         for cluster_name in model_cluster_names:
    #             df1 = pd.DataFrame(np.array([profile for _, profile in df.query(f'session == "{session}" & model_name == "{model_name}" & cluster_name == {cluster_name}')['profile'].iteritems()]))
    #             print(df1)
    #         #     plt.figure()
    #         #     plt.plot(x=df1.T.values, y=df1.T.index, alpha=0.3, color=cluster_colors[cluster_name])
    #         #     # df1.T.plot()
    #         #     # sns.lineplot(data=df1.T, color=cluster_colors[cluster_name], plot_kws=dict(alpha=0.3))
    #         #     plt.title(f'{bundle_name} {session} {model_name} fa profiles')
    #         #     plt.savefig(join(base_dir, f'{session}_{model_name}_cluster_{cluster_name}_profile.png'))
    #         #     plt.close()

    # 95% ci plot
    for session in session_names:
        session_model_names = df.model_name.unique()
        for model_name in session_model_names:
            model_cluster_names = df.query(f'session == "{session}" & model_name == "{model_name}"')['cluster_name'].unique()
            
            plt.figure()

            for cluster_name in model_cluster_names:
                df1 = pd.DataFrame(np.array([profile for _, profile in df.query(f'session == "{session}" & model_name == "{model_name}" & cluster_name == {cluster_name}')['profile'].iteritems()]))
                df2 = pd.melt(frame=df1, var_name='node', value_name='fa_value')
                sns.lineplot(data=df2, x='node', y='fa_value')

            plt.title(f'{bundle_name} {session} {model_name} cluster fa profiles')
            plt.savefig(join(base_dir, f'{session}_{model_name}_cluster_profile_ci.png'))
            plt.close()


def find_best_clusters(dice_matrix, dice_pairs=None):
    ''' recursive function to find the best clusters based on dice coefficient scores '''    
    import numpy as np

    if dice_pairs is None:
        dice_pairs = []

    # ensure search modifies a copy
    _dice_matrix = dice_matrix.copy()

    # find the maximum dice coeffient
    idx = np.unravel_index(np.argmax(_dice_matrix, axis=None), _dice_matrix.shape)
    dice_pairs.append(idx)

    # remove the row and column from future considersation;
    # setting to negative dice coefficient value
    _dice_matrix[idx[0], :] = -1
    _dice_matrix[:, idx[1]] = -1

    if (_dice_matrix == -1*np.ones(_dice_matrix.shape)).all():
        return dice_pairs
    else:
        return find_best_clusters(_dice_matrix, dice_pairs)


def number_of_models(model_names, subject, session):
    num_models = 0

    for model_name in model_names[subject][session]:
        num_models += 1

    return num_models


def number_of_total_clusters(cluster_names, subject, session):
    num_clusters = 0

    for names in cluster_names[subject][session]:
        num_clusters += len(names)

    return num_clusters


def number_of_clusters(cluster_names, subject, session, model_idx):
    num_clusters = []

    for names in cluster_names[subject][session]:
        num_clusters.append(len(names))

    return num_clusters[model_idx]


def population_visualizations(base_dir, bundle_name, bundle_dice_coef, cluster_dice_coef, bundle_profile_fa_r2, cluster_profile_fa_r2, model_names, cluster_names):
    """
    clusters dice and fa r2 bar chart
    """
    from os.path import join
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    summary_dfs = {}
    
    for subject in subjects:
        summary_dfs[subject] = {}
    
        df = pd.DataFrame(columns=["hyperparameter", "pair", "dice", "fa_r2"])

        # pad dataframe  - to visually space model clusters
        # df = df.append({
        #             "hyperparameter": '',
        #             "pair": '',
        #             "dice": '',
        #             "fa_r2": ''
        #         }, ignore_index=True)

        dice_matrix = cluster_dice_coef[subject]
        profile_matrix = cluster_profile_fa_r2[subject]

        for model_num in range(number_of_models(model_names, subject, session_names[1])):
            test_clusters = number_of_clusters(cluster_names, subject, session_names[0], model_num)
            retest_clusters = number_of_clusters(cluster_names, subject, session_names[1], model_num)

            num_clusters = np.amin([test_clusters, retest_clusters])

            i = model_num*num_clusters
            j = i+num_clusters
            ij_slice = np.s_[i:j,i:j]

            dice_pairs = find_best_clusters(dice_matrix[ij_slice])

            for dice_pair in dice_pairs:
                df = df.append({
                    "hyperparameter": f'{model_num}',
                    "pair": dice_pair,
                    "dice": dice_matrix[ij_slice][dice_pair],
                    "fa_r2": profile_matrix[ij_slice][dice_pair]
                }, ignore_index=True)

            # pad dataframe - to visually space model clusters
            # df = df.append({
            #         "hyperparameter": '',
            #         "pair": '',
            #         "dice": '',
            #         "fa_r2": ''
            #     }, ignore_index=True)

        # TODO could plot each subject
        df.to_csv(join(base_dir, f'{subject}_{bundle_name}_summary.csv'))
        summary_dfs[subject] = df
    
    frames = []
    for subject in subjects:
        df = summary_dfs[subject]
        df['subject'] = subject
        df.index.name='idx'
        frames.append(df)
    bundle_df = pd.concat(frames).groupby('idx').agg({'dice': ['mean', 'std'], 'fa_r2': ['mean', 'std']})
    bundle_df.to_csv(join(base_dir, f'{bundle_name}_summary.csv'))

    fig, ax = plt.subplots(figsize=(20,4))
    ax.set_ylabel('dice')

    ax2 = ax.twinx()
    ax2.set_ylabel('fa $r^2$')
    
    bundle_df.dice.plot(kind='bar', y='mean', ax=ax, color='tab:blue', edgecolor='k', label='mean dice', alpha=0.75, yerr='std', error_kw=dict(elinewidth=0.5), legend=False, width=0.25, position=1)
    bundle_df.fa_r2.plot(kind='bar', y='mean', ax=ax2, color='tab:orange', edgecolor='k', label='mean fa $r^2$', alpha=0.75, yerr='std', error_kw=dict(elinewidth=0.5),legend=False, width=0.25, position=0)

    # add bundle dice and fa r2 lines
    ax.hlines(np.mean(list(bundle_dice_coef.values())), 0, len(bundle_df.index), colors='tab:blue', linestyles='dashed')
    ax.hlines(np.mean(list(bundle_profile_fa_r2.values())), 0, len(bundle_df.index), colors='tab:orange', linestyles='dashed')

    # TODO x-axes for now not crictical
    
    ax.set_xlabel('cluster')
#     ax.set_xticklabels(getxticklabels(), rotation=0)

#     ax1 = ax.twiny()
#     ax1.xaxis.set_ticks_position('bottom')
#     ax1.spines["bottom"].set_position(("axes", -0.25))
#     ax1.set_xticks(range(len(df.index)))

#     ax1.xaxis.set_major_locator(FixedLocator(getmodelxticklocations()))
#     ax1.set_xticklabels(df['hyperparameter'].unique())
#     ax1.set_xlabel('hyperparameter group')
#     ax1.xaxis.set_label_position("bottom")
    
#     ax3 = ax.twiny()
#     ax3.xaxis.set_ticks_position('bottom')
#     ax3.spines["bottom"].set_position(("axes", -0.25))
#     ax3.set_xticks(range(len(bundle_df.index)))
# #     ax3.xaxis.set_major_locator(FixedLocator([i for i in range(len(df.index)) if i not in [i for i in range(len(df.index))[1:len(df.index):3]]]))
#     ax3.tick_params(labelbottom=False, direction='in') 

    fig.legend()
    plt.title(f'Test-Retest {bundle_name} (N={len(subjects)})')
    # plt.show()
    plt.savefig(join(base_dir, f'test-retest_model_comparision.png'))
    plt.close()
