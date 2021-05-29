"""
visualization utilities
"""
from subbundle_model_analysis_utils import ClusterType

def visualize_tractogram(sft, bundle_dict):
    """
    plotly visualzation for clusters using MNI space
    
    Parameters
    ----------
    sft : StatefulTractogram
    bundle_dict : dict
    """
    from AFQ.viz.plotly_backend import visualize_volume, visualize_bundles
    from AFQ.data import read_mni_template

    figure = visualize_volume(
        read_mni_template().get_fdata(),
        interact = False,
        inline = False
    )

    return visualize_bundles(sft, bundle_dict=bundle_dict, figure=figure)


def visualize_subject_clusters(subject, centroids, bundle_dict):
    """
    take the subject and show the centroid for each cluster
    
    Parameters
    ----------
    subject : string
    centroids : dict
    bundle_dict : dict
    """
    from dipy.io.stateful_tractogram import StatefulTractogram

    clusters = []
    for cluster_centroid in centroids[subject]:
        clusters.append(cluster_centroid.streamlines[0])
    
    sft = StatefulTractogram.from_sft(clusters, cluster_centroid)
    return visualize_tractogram(sft, bundle_dict)


def display_consensus_centroids(metadata, cluster_info):
    """
    for each `n_clusters` in expirement (i.e., RANGE_N_CLUSTERS)
    show the centroids for the consensus subject
    NOTE: that the consensus subject may differ across clusters
    """
    from IPython.display import display

    for n_clusters in metadata['experiment_range_n_clusters']:
        consensus_subject = cluster_info[n_clusters]['consensus_subject']
        for session_name in metadata['experiment_sessions']:
            print('n_clusters', n_clusters, 'session', session_name, 'consensus subject', consensus_subject)
            display(visualize_subject_clusters(
                consensus_subject,
                cluster_info[n_clusters][session_name]['centroids'],
                metadata['experiment_bundle_dict']
            ))


def display_subject_centriods(metadata, cluster_info, subject_id):
    """
    for each `n_clusters` in expirement (i.e., RANGE_N_CLUSTERS)
    show the centroids for the requested subject
    """
    from IPython.display import display

    for n_clusters in metadata['experiment_range_n_clusters']:
        for session_name in metadata['experiment_sessions']:
            print('n_clusters', n_clusters, 'session', session_name, 'subject', subject_id)
            display(visualize_subject_clusters(
                subject_id,
                cluster_info[n_clusters][session_name]['centroids'],
                metadata['experiment_bundle_dict']
            ))


def display_streamline_count_centroids(metadata, cluster_info):
    from IPython.display import display
    from identify_subbundles import convert_centroids

    for n_clusters in metadata['experiment_range_n_clusters']:
        for session_name in metadata['experiment_sessions']:
            print('n_clusters', n_clusters, 'session', session_name)
            mni_prealign_sft = convert_centroids(
                n_clusters,
                cluster_info[n_clusters][session_name]['centroids'],
                metadata['experiment_bundle_dict']
            )
            display(visualize_tractogram(mni_prealign_sft, metadata['experiment_bundle_dict']))


def display_centroids(metadata, cluster_info):
    from os.path import join
    from IPython.display import display
    from identify_subbundles import match_clusters, get_relabeled_centroids, convert_centroids

    base_dir = join(metadata['experiment_output_dir'], metadata['bundle_name'])

    for n_clusters in metadata['experiment_range_n_clusters']:
        for session_name in metadata['experiment_sessions']:
            print('n_clusters', n_clusters, 'session_name', session_name)
            match_clusters(
                base_dir,
                session_name,
                metadata['experiment_subjects'],
                cluster_info,
                cluster_info[n_clusters]['consensus_subject'],
                n_clusters,
                metadata['algorithm']
            )

            _mni_centroids = get_relabeled_centroids(metadata, n_clusters, session_name, cluster_info[n_clusters]['consensus_subject'])
        
            _mni_sft = convert_centroids(n_clusters, _mni_centroids, metadata['experiment_bundle_dict'])
            display(visualize_tractogram(_mni_sft, metadata['experiment_bundle_dict']))


def display_population_cluster_profile(metadata, cluster_profiles, scalar_name, n_clusters):
    """
    plot of test and retest mean afq profiles and confidence intervals per cluster in each model
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import math

    df = pd.DataFrame(columns=["session", "cluster_name", "subject", "profile"])

    max_value = {}

    for subject in metadata['experiment_subjects']:
        for session in metadata['experiment_sessions']:
            max_value[session] = 0

            for cluster_label in range(n_clusters):
                profile = cluster_profiles[subject][session][cluster_label]
                
                # find max value to set ylim
                profile_max_value = profile.max()
                if (profile_max_value > max_value[session]):
                    max_value[session] = profile_max_value
                
                df = df.append({
                    'session': session,
                    'cluster_label': cluster_label,
                    'subject': subject,
                    'profile': profile
                }, ignore_index=True)
    
    colors = sns.color_palette()
    
    for session in metadata['experiment_sessions']:
        print(session, n_clusters)
        plt.figure()

        for cluster_label in range(n_clusters):
            df1 = pd.DataFrame(np.array([profile for _, profile in df.query(f'session == "{session}" & cluster_label == {cluster_label}')['profile'].iteritems()]))
            df2 = pd.melt(frame=df1, var_name='node', value_name=scalar_name)
            sns.lineplot(data=df2, x='node', y=scalar_name, color=colors[cluster_label])

        plt.ylim(0., 10**int(math.log10(abs(max(max_value.values())))))
        plt.rc('axes', labelsize=14)
        plt.show()
        plt.close()


def display_population_cluster_profiles(metadata, cluster_afq_profiles):
    """
    display the scalar profiles for each `n_clusters`
    """
    import itertools

    for n_clusters, scalar in itertools.product(metadata['experiment_range_n_clusters'], metadata['model_scalars']):
        scalar_name = scalar.split('.')[0].replace('_', ' ')
        display_population_cluster_profile(
            metadata,
            cluster_afq_profiles[n_clusters][scalar],
            scalar_name,
            n_clusters
        )

def plot_silhouette_scores(embedding, cluster_labels):
    """
    see Selecting the number of clusters with silhouette analysis on KMeans clustering
    https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html

    e.g.
    embedding : np.load('mase_kmeans_fa_r2_is_mdf_embeddings_filtered.npy')
    cluster_labels : np.load('mase_kmeans_fa_r2_is_mdf_cluster_labels_filtered.npy')
    """
    import numpy as np
    from sklearn.metrics import silhouette_samples, silhouette_score
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import seaborn as sns

    n_clusters = len(np.unique(cluster_labels))

    average_silhouette_score = silhouette_score(embedding, cluster_labels)
    sample_silhouette_scores = silhouette_samples(embedding, cluster_labels)
    
    plt.figure()
    ax = plt.gca()
    # The silhouette scores can range from -1, 1
#         ax.set_xlim([-1,1])
    ax.set_xlim([-0.2,1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters
    ax.set_ylim([0, len(embedding) + (n_clusters + 1) * 10])
    
    y_lower = 10
    
    for i in range(n_clusters):
        ith_cluster_silhouette_scores = \
            sample_silhouette_scores[cluster_labels == i]

        ith_cluster_silhouette_scores.sort()

        size_cluster_i = ith_cluster_silhouette_scores.shape[0]
        
        y_upper = y_lower + size_cluster_i
        
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0, 
            ith_cluster_silhouette_scores,
            cmap = ListedColormap(sns.color_palette()),
            alpha=0.7
        )

        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax.set_title("The silhouette plot.")
    ax.set_xlabel("The silhouette scores")
    ax.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax.axvline(x=average_silhouette_score, color="red", linestyle="--")

    ax.set_yticks([])  # Clear the yaxis labels / ticks
#         ax.set_xticks(list(np.arange(-10, 12, 2)/10))
    ax.set_xticks(list(np.arange(0, 12, 2)/10))

    plt.show()
    plt.close()
    
    
def plot_pairplot(embedding, cluster_labels):
    """
    e.g.
    embedding : np.load('mase_kmeans_fa_r2_is_mdf_embeddings_filtered.npy')
    cluster_labels : np.load('mase_kmeans_fa_r2_is_mdf_cluster_labels_filtered.npy')
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    n_clusters = len(np.unique(cluster_labels))

    df = pd.DataFrame(embedding)
    df_labels = pd.DataFrame(cluster_labels, columns=['Type'])
    df = pd.concat([df_labels, df], axis=1)
    
    plt.figure()
    sns.pairplot(
        df,
        hue='Type',
        palette=sns.color_palette()[:n_clusters]
    )
    
    plt.show()
    plt.close()


def plot_profile(streamline_profile, bundle_profile, cluster_profiles):
    """
    streamline_profile : np.load('streamline_profile_fa.npy')
    bundle_profile : np.load('bundle_profile_fa.npy')
    cluster_profiles : [np.load('cluster_0_profile_fa.npy') ... ]

    TODO: could create a wrapper to plot subject, session, n_clusters
    TODO: could order cluster_profiles by relabeling algorithm
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    pd.DataFrame(streamline_profile.T).plot(color='blue', alpha=0.01, legend=False)
    pd.Series(bundle_profile, name='bundle').plot(color='black')

    cluster_label = 0
    for cluster_profile in cluster_profiles:
        pd.Series(cluster_profile, name=f'cluster {cluster_label}').plot(color=sns.color_palette()[cluster_label])
        cluster_label += 1

    plt.show()
    plt.close()


def get_bundle_anatomy_figures(metadata, model_data, subject):
    """
    TODO: plot appropriate bundle color
    """
    from IPython.display import display
    from AFQ.api import make_bundle_dict
    from identify_subbundles import move_tractogram_to_MNI_space

    bundle_anatomy_figures = {}

    for session_name in metadata['experiment_sessions']:
        
        print(metadata['bundle_name'], subject, session_name)
        sft = model_data[metadata['bundle_name']]['bundle_tractograms'][subject][session_name]
        sft = move_tractogram_to_MNI_space(session_name, subject, sft)
        bundle_anatomy_figures[session_name] = visualize_tractogram(sft, make_bundle_dict())

    return bundle_anatomy_figures

def add_colorbar(im, aspect=5, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    import matplotlib.pyplot as plt
    from mpl_toolkits import axes_grid1
    
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


def plot_adjacencies(metadata, model_data, subject):
    import matplotlib.pyplot as plt
    
    for session_name in metadata['experiment_sessions']:
        print(metadata['bundle_name'], subject, session_name)
        adjacencies = model_data[metadata['bundle_name']]['adjacencies'][subject][session_name]

        for adjacency in adjacencies:
            im = plt.imshow(adjacency, cmap='hot', interpolation='nearest')
            add_colorbar(im)
            plt.show()
            plt.close()


def plot_artifacts(metadata, model_data, subject, n_clusters):
    for session_name in metadata['experiment_sessions']:
        print(metadata['bundle_name'], n_clusters, subject, session_name)
        filtered_embeddings = model_data[metadata['bundle_name']]['filtered_embeddings'][subject][session_name][n_clusters]
        filtered_cluster_labels = model_data[metadata['bundle_name']]['filtered_cluster_labels'][subject][session_name][n_clusters]

        plot_silhouette_scores(filtered_embeddings, filtered_cluster_labels)
        plot_pairplot(filtered_embeddings, filtered_cluster_labels)


def plot_scalar_profiles(metadata, model_data, subject, n_clusters):
    for session_name in metadata['experiment_sessions']:
        print(metadata['bundle_name'], n_clusters, subject, session_name, 'FA')
        streamline_profile = model_data[metadata['bundle_name']]['streamline_profiles'][subject][session_name]
        bundle_profile = model_data[metadata['bundle_name']]['bundle_profiles'][subject][session_name]
        cluster_profiles = model_data[metadata['bundle_name']]['cluster_profiles'][subject][session_name][n_clusters]
        plot_profile(streamline_profile, bundle_profile, cluster_profiles)


def plot_cluster_streamlines(metadata, model_data, subject, n_clusters):
    """
    TODO plot all in one
    """
    from IPython.display import display
    from identify_subbundles import move_tractogram_to_MNI_space

    for session_name in metadata['experiment_sessions']:
        print(metadata['bundle_name'], n_clusters, subject, session_name)
        sfts = model_data[metadata['bundle_name']]['clean_cluster_tractograms'][subject][session_name][n_clusters]

        for sft in sfts:
            sft = move_tractogram_to_MNI_space(session_name, subject, sft)
            display(visualize_tractogram(sft, metadata['experiment_bundle_dict']))


def show_choose_k_data(data):
    """
    Little visualization to show average root mean squared error across test-retest
    """
    from IPython.display import display
    import matplotlib.pyplot as plt
    import pandas as pd
    
    n_clusters = len(data)
    Ks = [k+1 for k in list(range(n_clusters))]
    df = pd.DataFrame(data, index=Ks)
#     display(df)

    # per subject avgRMSE
    plt.figure()
    ax = df.plot(legend=False)
    ax.locator_params(integer=True)
    plt.show()
    # display(ax)
    
    # population average avgRMSE
    plt.figure()
    ax = df.mean(axis=1).plot(legend=False, color='black')
    ax.locator_params(integer=True)
    plt.show()
    # display(ax)


def display_population_bundle_profiles(metadata, bundle_profiles, scalar_name='DTI FA'):
    """
    plots the mean and 95 percent confidence interval population bundle profiles for each session
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    
    df = pd.DataFrame(columns=["session", "subject", "profile"])
    
    for subject in metadata['experiment_subjects']:
        for session in metadata['experiment_sessions']:
            profile = bundle_profiles[subject][session]
            df = df.append({
                'session': session,
                'subject': subject,
                'profile': profile
            }, ignore_index=True)
            
    
    for session in metadata['experiment_sessions']:
        print(session)
        plt.figure()

        df1 = pd.DataFrame(np.array([profile for _, profile in df.query(f'session == "{session}"')['profile'].iteritems()]))
        df2 = pd.melt(frame=df1, var_name='node', value_name=scalar_name)
        sns.lineplot(data=df2, x='node', y=scalar_name, color='black')
        plt.rc('axes', labelsize=14)
        plt.show()
        plt.close()


def get_bundle_streamline_count(metadata, model_data, subject):
    """
    For `subject` gets streamline counts for both test and retest sessions

    Returns
    -------
    DataFrame : one row for `subject` with two columns for `sessions`
    """
    import pandas as pd
    
    data = {}
    for session_name in metadata['experiment_sessions']:
        data[session_name] = [len(model_data[metadata['bundle_name']]['bundle_tractograms'][subject][session_name])]

    return pd.DataFrame(data, index=[subject])

def get_consensus_streamline_counts(metadata, model_data, cluster_info):
    """
    For each `n_clusters` gets the `consensus_subject` streamline counts for both test and retest sessions

    Returns
    -------
    DataFrame : with row for each `consensus_subject` with two columns for `sessions`
    """
    import pandas as pd
    
    dfs = []
    
    for n_clusters in metadata['experiment_range_n_clusters']:
        df = get_bundle_streamline_count(metadata, model_data, cluster_info[n_clusters]['consensus_subject'])
        df.insert(0, 'n_clusters', n_clusters)
        dfs.append(df)
        
    return pd.concat(dfs)

def get_bundle_streamline_counts(metadata, model_data):
    """
    Returns
    -------
    DataFrame : with row for each `subject` with two columns for `sessions`
    """
    import pandas as pd
    
    dfs = []
    
    for subject in metadata['experiment_subjects']:
        dfs.append(get_bundle_streamline_count(metadata, model_data, subject))
    
    return pd.concat(dfs)

def get_bundle_streamline_stats(metadata, model_data):
    """
    Returns
    -------
    DataFrame : descriptive statistics for population
    """
    
    df = get_bundle_streamline_counts(metadata, model_data)

    return df.describe()


def display_bundle_streamline_stats(metadata, model_data):
    from IPython.display import display
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    bss = get_bundle_streamline_stats(
        metadata, model_data
    )

    display(
        bss.style.set_caption(f"{metadata['bundle_name']} streamline count statistics")
    )

    sns.violinplot(data=get_bundle_streamline_counts(metadata, model_data), color='gray')
    plt.show()
    
    # bar plot
#     bss.loc['mean'].plot(kind='bar', color='gray', yerr=bss.loc['std'])
#     plt.show()


def get_bundle_dice_coeff_stats(bundle_dice_coeffs):
    """
    Returns
    -------
    DataFrame : descriptive statistics for population
    """
    import pandas as pd

    df = pd.DataFrame(bundle_dice_coeffs, index=['weighted dice coeff'])

    return df.T.describe()


def display_bundle_dice_coeff_stats(metadata, bundle_dice_coeffs):
    from IPython.display import display
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    bdcs = get_bundle_dice_coeff_stats(
        bundle_dice_coeffs
    )

    display(
        bdcs.style.set_caption(f"{metadata['bundle_name']} bundle weighted dice coefficient statistics")
    )
    
    df = pd.DataFrame(bundle_dice_coeffs, index=['weighted dice coeff'])
    sns.violinplot(data=df.T, color='gray')
    plt.show()

#     bdcs.loc['mean'].plot(kind='bar', color='gray', yerr=bdcs.loc['std'])
#     plt.show()
    

def get_bundle_profile_reliability(metadata, bundle_profiles, scalar_name='DTI FA'):
    import numpy as np
    import pandas as pd
    
    test_retest_bundle_profile_corr = {}
    
    for subject in metadata['experiment_subjects']:
        test_fa = bundle_profiles[subject][metadata['experiment_sessions'][0]]
        retest_fa = bundle_profiles[subject][metadata['experiment_sessions'][1]]
        
        test_retest_bundle_profile_corr[subject] = np.corrcoef(test_fa, retest_fa)[0, 1]

    return pd.DataFrame(test_retest_bundle_profile_corr, index=[f'{scalar_name} pearson r'])

def get_bundle_profile_reliability_stats(metadata, bundle_profiles, scalar_name='DTI FA'):
    """
    Returns
    -------
    DataFrame : descriptive statistics for population
    """

    df = get_bundle_profile_reliability(metadata, bundle_profiles, scalar_name)

    return df.T.describe()


def display_bundle_profile_reliability_stats(metadata, bundle_profiles, scalar_name='DTA FA'):
    from IPython.display import display
    import seaborn as sns
    import matplotlib.pyplot as plt

    bprs = get_bundle_profile_reliability_stats(
        metadata,
        bundle_profiles
    )
    
    display(
        bprs.style.set_caption(f"{metadata['bundle_name']} {scalar_name} pearsons r statistics")
    )

    df = get_bundle_profile_reliability(metadata, bundle_profiles, scalar_name)
    sns.violinplot(data=df.T, color='gray')
    plt.show()
    

#     bprs.loc['mean'].plot(kind='bar', color='gray', yerr=bprs.loc['std'])
#     plt.show()
    
def get_cluster_streamline_counts(metadata, model_data, cluster_type=ClusterType.CLEAN):
    import pandas as pd
    
    dfs = []

    for n_clusters in metadata['experiment_range_n_clusters']:
        data = {}

        for subject in metadata['experiment_subjects']:
            data[subject] = {}
            for session in metadata['experiment_sessions']:
                for cluster_id in range(n_clusters):
                    sft = model_data[metadata['bundle_name']][f'{cluster_type}_cluster_tractograms'][subject][session][n_clusters][cluster_id]
                    data[subject][session + ' cluster ' + str(cluster_id)] = len(sft)


        dfs.append(pd.DataFrame(data).T)
        
    return dfs


def display_cluster_streamline_count_stats(metadata, csc, n_clusters):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from IPython.display import display

    idx = metadata['experiment_range_n_clusters'].index(n_clusters)
    
    cscs = csc[idx].describe()
    
    display(
        cscs.style.set_caption(f"{metadata['bundle_name']} K={n_clusters} cluster streamline count statistics")
    )

    ax = sns.violinplot(data=csc[idx], palette=sns.color_palette(as_cmap=True)[0:n_clusters]*2)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    plt.show()

#     colors = sns.color_palette()
#     cscs.loc['mean'].plot(kind='bar', color=colors[0:n_clusters]*2, yerr=cscs.loc['std'])
#     plt.show()


def get_cluster_dice_coeff_stats(cluster_dice_coeffs, n_clusters):
    import pandas as pd
    # from IPython.display import display

    cdc = pd.DataFrame(cluster_dice_coeffs[n_clusters], index=list(range(n_clusters)))
    
    # TODO noticed some low overlaps in the first bundle
    # which is unexpected since supposedly maximizing trace of weighted dice.
    # display(cdc)
    
    return cdc.T.describe()


def display_cluster_dice_coef(metadata, cluster_dice_coeffs, n_clusters, bundle_mean):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from IPython.display import display
    import pandas as pd

    colors = sns.color_palette()

    cdcs = get_cluster_dice_coeff_stats(cluster_dice_coeffs, n_clusters)
    
    display(
        cdcs.style.set_caption(f"{metadata['bundle_name']} K={n_clusters} cluster weighted dice coefficient statistics")
    )

    cdc = pd.DataFrame(cluster_dice_coeffs[n_clusters], index=list(range(n_clusters)))
    ax = sns.violinplot(data=cdc.T, palette=sns.color_palette(as_cmap=True)[0:n_clusters])
    ax.axhline(y=bundle_mean, color="red", linestyle="--")
    plt.show()

    # print(f"{metadata['bundle_name']} K={n_clusters}")
    # ax = cdcs.loc['mean'].plot(kind="bar", color=colors, yerr=cdcs.loc['std'])
    # ax.axhline(y=bundle_mean, color="red", linestyle="--")
    # plt.show()


def get_cluster_profile_reliability(metadata, cluster_afq_profiles):
    import pandas as pd
    import numpy as np

    dfs = []

    for n_clusters in metadata['experiment_range_n_clusters']:
        fa_profiles = cluster_afq_profiles[n_clusters][metadata['model_scalars'][0]]
        test_retest_cluster_profile_r2 = {}

        for subject in metadata['experiment_subjects']:
            test_retest_cluster_profile_r2[subject] = {}
            subject_fa_profiles = fa_profiles[subject]

            for cluster_id in range(n_clusters):
                test_fa = subject_fa_profiles[metadata['experiment_sessions'][0]][cluster_id]
                retest_fa = subject_fa_profiles[metadata['experiment_sessions'][1]][cluster_id]

                test_retest_cluster_profile_r2[subject][cluster_id] = np.corrcoef(test_fa,retest_fa)[0, 1]
                # check pearsonr and np.corrcoef same thing
                # print(np.corrcoef(test_fa,retest_fa)[0, 1], pearsonr(test_fa,retest_fa))

        dfs.append(pd.DataFrame(test_retest_cluster_profile_r2, index=list(range(n_clusters))))

    return dfs


def display_cluster_profile_reliability_stats(metadata, cpr, n_clusters, bundle_mean):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from IPython.display import display
    
    colors = sns.color_palette()

    idx = metadata['experiment_range_n_clusters'].index(n_clusters)
    
    cprs = cpr[idx].T.describe()
    
    display(
        cprs.style.set_caption(f"{metadata['bundle_name']} K={n_clusters} DTI FA pearson r statistics")
    )
    
    ax = sns.violinplot(data=cpr[idx].T, palette=sns.color_palette(as_cmap=True)[0:n_clusters])
    ax.axhline(y=bundle_mean, color="red", linestyle="--")
    plt.show()

#     print(f"{metadata['bundle_name']} K={n_clusters}")
#     ax = cprs.loc['mean'].plot(kind='bar', color=colors, yerr=cprs.loc['std'])
#     ax.axhline(y=bundle_mean, color="red", linestyle="--")
#     plt.show()


def get_consensus_bundle_anatomy_figures(metadata, model_data, cluster_info):
    cluster_figures = {}
    for n_clusters in metadata['experiment_range_n_clusters']:
        consensus_subject = cluster_info[n_clusters]['consensus_subject']
        cluster_figures[n_clusters] = get_bundle_anatomy_figures(metadata, model_data, consensus_subject)
    return cluster_figures

def get_consensus_bundle_anatomy_figures(metadata, model_data, cluster_info, n_clusters):
    consensus_subject = cluster_info[n_clusters]['consensus_subject']
    return get_bundle_anatomy_figures(metadata, model_data, consensus_subject)


def get_bundle_dice_coeff(bundle_dice_coeffs, subject):
    import pandas as pd
    
    return pd.DataFrame({'weighted dice coeff': bundle_dice_coeffs[subject]}, index=[subject])

def get_consensus_bundle_dice_coeff(metadata, bundle_dice_coeffs, cluster_info):
    import pandas as pd
    
    dfs = []
    for n_clusters in metadata['experiment_range_n_clusters']:
        df = get_bundle_dice_coeff(bundle_dice_coeffs, cluster_info[n_clusters]['consensus_subject'])
        df.insert(0, 'n_clusters', n_clusters)
        dfs.append(df)

    return pd.concat(dfs)

def get_consensus_bundle_profile_reliability(metadata, bundle_profiles, cluster_info, scalar_name='DTI FA'):
    import pandas as pd

    dfs = []

    for n_clusters in metadata['experiment_range_n_clusters']:
        dfs.append(get_consensus_bundle_profile_reliability(metadata, bundle_profiles, cluster_info, n_clusters, scalar_name))

    return pd.concat(dfs)

def get_consensus_bundle_profile_reliability(metadata, bundle_profiles, cluster_info, n_clusters, scalar_name='DTI FA'):
    import numpy as np
    import pandas as pd

    consensus_subject = cluster_info[n_clusters]['consensus_subject']
    test_fa = bundle_profiles[consensus_subject][metadata['experiment_sessions'][0]]
    retest_fa = bundle_profiles[consensus_subject][metadata['experiment_sessions'][1]]
    
    return pd.DataFrame({'n_clusters': n_clusters, f'{scalar_name} pearson r': np.corrcoef(test_fa, retest_fa)[0, 1]}, index=[consensus_subject])

def display_consensus_adjacencies(metadata, model_data, cluster_info):
    for n_clusters in metadata['experiment_range_n_clusters']:
        display_consensus_adjacencies(metadata, model_data, cluster_info, n_clusters)

def display_consensus_adjacencies(metadata, model_data, cluster_info, n_clusters):
    consensus_subject = cluster_info[n_clusters]['consensus_subject']
    plot_adjacencies(metadata, model_data, consensus_subject)

def display_consensus_model_artifacts(metadata, model_data, cluster_info):
    for n_clusters in metadata['experiment_range_n_clusters']:
        display_consensus_model_artifacts(metadata, model_data, cluster_info, n_clusters)

def display_consensus_model_artifacts(metadata, model_data, cluster_info, n_clusters):
    for session_name in metadata['experiment_sessions']:
        consensus_subject = cluster_info[n_clusters]['consensus_subject']
        print(metadata['bundle_name'], n_clusters, consensus_subject, session_name)
        embeddings = model_data[metadata['bundle_name']]['embeddings'][consensus_subject][session_name][n_clusters]
        cluster_labels = model_data[metadata['bundle_name']]['cluster_labels'][consensus_subject][session_name][n_clusters]

        plot_silhouette_scores(embeddings, cluster_labels)
        plot_pairplot(embeddings, cluster_labels)

def display_consensus_filtered_artifacts(metadata, model_data, cluster_info):
    for n_clusters in metadata['experiment_range_n_clusters']:
        display_consensus_filtered_artifacts(metadata, model_data, cluster_info, n_clusters)

def display_consensus_filtered_artifacts(metadata, model_data, cluster_info, n_clusters):
    consensus_subject = cluster_info[n_clusters]['consensus_subject']
    plot_artifacts(metadata, model_data, consensus_subject, n_clusters)

def display_streamline_bundle_profile(metadata, model_data, subject, session_name):
    """
    plot streamline and bundle profiles
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    print(metadata['bundle_name'], subject, session_name)
    
    streamline_profile = model_data[metadata['bundle_name']]['streamline_profiles'][subject][session_name]
    bundle_profile = model_data[metadata['bundle_name']]['bundle_profiles'][subject][session_name]
    
    pd.DataFrame(streamline_profile.T).plot(color='blue', alpha=0.01, legend=False)
    pd.Series(bundle_profile, name='bundle').plot(color='black')
    
    plt.show()
    plt.close()
    
def display_consensus_streamline_bundle_profile(metadata, model_data, cluster_info):
    for n_clusters in metadata['experiment_range_n_clusters']:
        display_consensus_streamline_bundle_profile(metadata, model_data, cluster_info, n_clusters)

def display_consensus_streamline_bundle_profile(metadata, model_data, cluster_info, n_clusters):
    consensus_subject = cluster_info[n_clusters]['consensus_subject']
    for session in metadata['experiment_sessions']:
        display_streamline_bundle_profile(metadata, model_data, consensus_subject, session)

def display_cluster_profile(metadata, model_data, subject, session_name, n_clusters):
    """
    plot cluster profiles
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    print(metadata['bundle_name'], n_clusters, subject, session_name)
    
    cluster_profiles = model_data[metadata['bundle_name']]['cluster_profiles'][subject][session_name][n_clusters]
    
    cluster_label = 0
    for cluster_profile in cluster_profiles:
        pd.Series(cluster_profile, name=f'cluster {cluster_label}').plot(color=sns.color_palette()[cluster_label])
        cluster_label += 1
    
    plt.show()
    plt.close()

def display_consensus_cluster_profiles(metadata, model_data, cluster_info):
    for n_clusters in metadata['experiment_range_n_clusters']:
        display_consensus_cluster_profiles(metadata, model_data, cluster_info, n_clusters)


def display_consensus_cluster_profiles(metadata, model_data, cluster_info, n_clusters):
    consensus_subject = cluster_info[n_clusters]['consensus_subject']
    for session in metadata['experiment_sessions']:
        display_cluster_profile(metadata, model_data, consensus_subject, session, n_clusters)


def get_cluster_tractogram_figure(metadata, model_data, cluster_type, subject, session_name, n_clusters):
    """
    combine cluster tractograms into single plot
    """
    from AFQ.utils.streamlines import bundles_to_tgram
    from AFQ.data import read_mni_template
    from identify_subbundles import move_tractogram_to_MNI_space

    bundle_dict = metadata['experiment_bundle_dict']
    
    sfts = model_data[metadata['bundle_name']][f'{cluster_type}_cluster_tractograms'][subject][session_name][n_clusters]

    bundles = {}
    cluster_id = 0
    for bundle_name in bundle_dict.keys():
        bundles[bundle_name] = move_tractogram_to_MNI_space(session_name, subject, sfts[cluster_id])
        cluster_id += 1

        if cluster_id == n_clusters:
            break

    return visualize_tractogram(bundles_to_tgram(bundles, bundle_dict, read_mni_template()), bundle_dict)

def get_clean_consensus_cluster_tractograms(metadata, model_data, cluster_info, n_clusters):
    clean_consensus_cluster_figs = {}

    for session_name in metadata['experiment_sessions']:
        clean_consensus_cluster_figs[session_name] = get_cluster_tractogram_figure(
            metadata, 
            model_data, 
            ClusterType.CLEAN, 
            cluster_info[n_clusters]['consensus_subject'], 
            session_name, 
            n_clusters
        )

    return clean_consensus_cluster_figs

def get_consensus_cluster_tractograms(metadata, model_data, cluster_info):
    consensus_cluster_figs = {}
    for cluster_type in [ClusterType.MODEL, ClusterType.FILTERED, ClusterType.CLEAN]:
        consensus_cluster_figs[cluster_type] = {}
        for n_clusters in metadata['experiment_range_n_clusters']:
            consensus_cluster_figs[cluster_type][n_clusters] = {}
            consensus_subject = cluster_info[n_clusters]['consensus_subject']
            for session_name in metadata['experiment_sessions']:
                consensus_cluster_figs[cluster_type][n_clusters][session_name] = get_cluster_tractogram_figure(
                    metadata, model_data, cluster_type, consensus_subject, session_name, n_clusters
                )
                
    return consensus_cluster_figs


def get_subject_cluster_streamline_counts(metadata, model_data, subject, n_clusters):
    """
    calculate the streamlines for each stage of cleaning process for both sessions
    """
    import pandas as pd
    
    data = {}
    
    for cluster_id in range(n_clusters):
        data[f'cluster {cluster_id}'] = {}
        for session_name in metadata['experiment_sessions']:
            data[f'cluster {cluster_id}'][session_name] = {}
            for cluster_type in [ClusterType.MODEL, ClusterType.FILTERED, ClusterType.CLEAN]:
                data[f'cluster {cluster_id}'][session_name][cluster_type] = {}
                sft = model_data[metadata['bundle_name']][f'{cluster_type}_cluster_tractograms'][subject][session_name][n_clusters][cluster_id]
                data[f'cluster {cluster_id}'][session_name][cluster_type] = len(sft)

    dfs = []
    keys = []
    
    for cluster_id in range(n_clusters):
        dfs.append(pd.DataFrame(data[f'cluster {cluster_id}']))
        keys.append(f'cluster {cluster_id}')
                   
    return pd.concat(dfs, keys=keys).T


def display_consensus_cluster_streamline_counts(metadata, model_data, cluster_info):
    for n_clusters in metadata['experiment_range_n_clusters']:
        display_consensus_cluster_streamline_counts(metadata, model_data, cluster_info, n_clusters)


def display_consensus_cluster_streamline_counts(metadata, model_data, cluster_info, n_clusters):
    from IPython.display import display

    consensus_subject = cluster_info[n_clusters]['consensus_subject']
    
    df = get_subject_cluster_streamline_counts(metadata, model_data, consensus_subject, n_clusters)
    
    display(
        df.style.set_caption(f"{metadata['bundle_name']} K={n_clusters} {consensus_subject} streamline counts")
    )


def get_cluster_dice_coef(cluster_dice_coeffs, subject, n_clusters):
    import pandas as pd
    
    return pd.DataFrame([cluster_dice_coeffs[n_clusters][subject]], index=[subject])

def display_consensus_cluster_dice_coef(metadata, cluster_info, cluster_dice_coeffs):
    for n_clusters in metadata['experiment_range_n_clusters']:
        display_consensus_cluster_dice_coef(metadata, cluster_info, cluster_dice_coeffs, n_clusters)

def display_consensus_cluster_dice_coef(metadata, cluster_info, cluster_dice_coeffs, n_clusters):
    from IPython.display import display

    consensus_subject = cluster_info[n_clusters]['consensus_subject']
    
    ccdc = get_cluster_dice_coef(
        cluster_dice_coeffs, consensus_subject, n_clusters
    )

    display(
        ccdc.style.set_caption(f"{metadata['bundle_name']} K={n_clusters} {consensus_subject} cluster weighted dice coefficient")
    )

def get_subject_cluster_profile_reliability(metadata, cluster_afq_profiles, subject, n_clusters):
    import pandas as pd
    import numpy as np
    
    test_retest_cluster_profile_r2 = {}
    
    subject_fa_profiles = cluster_afq_profiles[n_clusters][metadata['model_scalars'][0]][subject]
    for cluster_id in range(n_clusters):
        test_fa = subject_fa_profiles[metadata['experiment_sessions'][0]][cluster_id]
        retest_fa = subject_fa_profiles[metadata['experiment_sessions'][1]][cluster_id]
                
        test_retest_cluster_profile_r2[cluster_id] = np.corrcoef(test_fa,retest_fa)[0, 1]

    return pd.DataFrame(test_retest_cluster_profile_r2, index=[subject])

def display_consensus_cluster_profile_reliability(metadata, cluster_info, cluster_afq_profiles):
    for n_clusters in metadata['experiment_range_n_clusters']:
        display_consensus_cluster_profile_reliability(metadata, cluster_info, cluster_afq_profiles, n_clusters)

def display_consensus_cluster_profile_reliability(metadata, cluster_info, cluster_afq_profiles, n_clusters):
    from IPython.display import display

    consensus_subject = cluster_info[n_clusters]['consensus_subject']
    
    ccpr = get_subject_cluster_profile_reliability(
        metadata, cluster_afq_profiles, consensus_subject, n_clusters
    )
    display(
        ccpr.style.set_caption(f"{metadata['bundle_name']} K={n_clusters} {consensus_subject} cluster DTI FA pearson r")
    )