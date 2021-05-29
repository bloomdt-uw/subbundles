"""
utility functions to support subbundle model analysis
"""
import logging

from numpy.core.fromnumeric import product
logger = logging.getLogger('subbundle')

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

def _download_scalar_data(scalar, base_dir, subjects, session_names, use_csd=True):
    """
    Download scalar data for `scalar` for all subjects and sessions from the
    single shell HCP reliability study.

        `{expirement_name}/{bundle_name}/{subject}/{session}/{scalar}`
    
    By default will download the CSD scalar data.

    NOTE: should download from hcp-subbundle bucket to ensure using same files

    Parameters
    ----------
    scalar_name : str
        either 'DTI_FA.nii.gz' or 'DTI_MD.nii.gz'
    base_dir : str
    subjects : list
    session_names : list
    use_csd : bool
        default True

    Returns
    -------
    scalar_data : dict
        dictionary with `subject` as key containing
        └── dictionary with `session_name` as key
            └── containing the scalar image fdata for that subject and session.
    """
    import s3fs
    from os.path import exists, join
    import nibabel as nib

    if use_csd:
        csd_suffix = '_CSD'
    else:
        csd_suffix = ''

    fs = s3fs.S3FileSystem()

    scalar_data = {}

    for subject in subjects:
        scalar_data[subject] = {}
        for session in session_names:
            scalar_filename = join(base_dir, subject, session, scalar)
            if not exists(scalar_filename):
                logger.log(logging.DEBUG, f'downloading {scalar_filename}')
                fs.get(
                    (
                        f'profile-hcp-west/hcp_reliability/single_shell/'
                        f'{session.lower()}_afq{csd_suffix}/'
                        f'sub-{subject}/ses-01/'
                        f'sub-{subject}_dwi_model-{scalar}'
                    ),
                    scalar_filename
                )

            logger.log(logging.DEBUG, f'loading {scalar_filename}')
            scalar_data[subject][session] = nib.load(scalar_filename).get_fdata()

    return scalar_data


def _download_adjacencies(expirement_name, base_dir, subjects, session_names, bundle_name, range_n_clusters):
    """
    Download all adjacency files used in constructing the model. Only used for visualization purposes.

        `{expirement_name}/{bundle_name}/{subject}/{session}/{n_clusters}/adjacency_*.npy`

    Parameters
    ----------
    expirement_name : str
    base_dir : str
    subjects : list
    session_names : list
    bundle_name : str
    range_n_clusters: list

    Returns
    -------
    adjacencies : dict
        dictionary with `subject` as key containing
        └── dictionary with `session_name` as key
            └── containing list of ndarray for each pairwise adjacency measure
                as indicated in file name
    """

    import s3fs
    from os.path import basename, exists, join
    import numpy as np

    fs = s3fs.S3FileSystem()

    adjacencies = {}

    for subject in subjects:
        adjacencies[subject] = {}

        for session in session_names:
            adjacencies[subject][session] = []

            # NOTE: these are the same across clusters so only need to download one
            n_clusters = range_n_clusters[0]
            remote_adjacencies = fs.glob(f'hcp-subbundle/{expirement_name}/{session}/{bundle_name}/{subject}/{n_clusters}/adjacency_*.npy')

            for remote_adjacency in remote_adjacencies:
                local_adjacency = join(base_dir, subject, session, basename(remote_adjacency))
                
                if not exists(local_adjacency):
                    logger.log(logging.DEBUG, f'downloading {local_adjacency}')
                    fs.get(remote_adjacency, local_adjacency)
                
                adjacencies[subject][session].append(np.load(local_adjacency))

    return adjacencies


def _download_bundle_tractograms(base_dir, subjects, session_names, bundle_name, use_csd=True, use_clean=True):
    """
    Download the bundle tractogram for all subjects and sessions from the 
    single shell HCP reliabilty study.

        `{expirement_name}/{bundle_name}/{subject}/{session}/{bundle_name}.trk`
    
    By default will download clean CSD tractograms.

    NOTE: should download from hcp-subbundle bucket to ensure using same files

    Parameters
    ----------
    base_dir : str
    subjects : list
    session_names : list
    use_csd : bool
        default True
    clean : bool
        default True

    Returns
    -------
    tractograms : dict
        dictionary with `subject` as key containing
        └── dictionary with `session_name` as key
            └── containing the bundle `StatefulTractogram` for that subject and session.
    """
    import s3fs
    from os.path import exists, join
    from dipy.io.streamline import load_tractogram

    if use_csd:
        csd_suffix = '_CSD'
    else:
        csd_suffix = ''
    
    if use_clean:
        clean_prefix = 'clean_'
    else:
        clean_prefix = ''

    fs = s3fs.S3FileSystem()

    tractogram_basename = f'{bundle_name}.trk'

    tractograms = {}
    
    for subject in subjects:
        tractograms[subject] = {}

        for session in session_names:
            tractogram_filename = join(base_dir, subject, session, tractogram_basename)

            if not exists(tractogram_filename):
                logger.log(logging.DEBUG, f'downloading {tractogram_filename}')
                fs.get(
                    (
                        f'profile-hcp-west/hcp_reliability/single_shell/'
                        f'{session.lower()}_afq{csd_suffix}/'
                        f'sub-{subject}/ses-01/'
                        f'{clean_prefix}bundles/sub-{subject}_dwi_space-RASMM_model-CSD_desc-det-afq-{bundle_name}_tractography.trk'
                    ),
                    tractogram_filename
                )

            logger.log(logging.DEBUG, f'loading {tractogram_filename}')
            tractogram = load_tractogram(tractogram_filename, 'same')
            tractograms[subject][session] = tractogram

    return tractograms

class ArtifactType:
    """
    enum representing supported cluster artifacts
    """
    EMBEDDINGS = 'embeddings'
    EMBEDDINGS_FILTERED = 'embeddings_filtered'
    CLUSTER_LABELS = 'cluster_labels'
    CLUSTER_LABELS_FILTERED = 'cluster_labels_filtered'

def _download_cluster_artifacts(expirement_name, base_dir, subjects, session_names, bundle_name, range_n_clusters, artifact_type):
    """
    Download cluster artifacts used in constructing the cluster tractograms.
    
    Only used for quality control purposes in pair plot and silhouette score visualizations.

    Embeddings are joint adjacency spectral embeddings, constructed from the adjacencies, and
    used in model as features.

    Cluster labels are the result of clustering. They specify which streamlines belong to which
    cluster.

    The filtered postfix are the main artifiacts of interest and signify that the artifacts 
    are after removing streamlines that are below the average silhouette score. 
    
    The no postfixed versions are the originals, and can be useful for quality control.

        `{expirement_name}/{session}/{bundle_name}/{subject}/{n_clusters}/*_{artifact_type}.npy`

    Parameters
    ----------
    expirement_name : str
    base_dir : str
    subjects : list
    session_names : list
    bundle_name : str
    range_n_clusters: list
    artifact_type : str

    Returns
    -------
    artifacts : dict
        dictionary with `subject` as key containing
        └── dictionary with `session_name` as key
            └── dictionary with `n_clusters` as key
                └── containing ndarray of the cluster artifact
    """
    import s3fs
    from os.path import basename, exists, join
    import numpy as np

    fs = s3fs.S3FileSystem()

    artifacts = {}

    for subject in subjects:
        artifacts[subject] = {}

        for session in session_names:
            artifacts[subject][session] = {}

            for n_clusters in range_n_clusters:
                artifacts[subject][session][n_clusters] = {}

                remote_artifact = fs.glob(f'hcp-subbundle/{expirement_name}/{session}/{bundle_name}/{subject}/{n_clusters}/*_{artifact_type}.npy')[0]

                local_artifact = join(base_dir, subject, session, str(n_clusters), basename(remote_artifact))
                
                if not exists(local_artifact):
                    logger.log(logging.DEBUG, f'downloading {local_artifact}')
                    fs.get(remote_artifact, local_artifact)
                
                artifacts[subject][session][n_clusters] = np.load(local_artifact)


    return artifacts

class ClusterType:
    """
    enum representing supported cluster tractogram types

    MODEL is the original cluster
    FILTERED is removing streamlines from MODEL with below average silhouette scores
    CLEAN is cleaning FILTERED using pyAFQ and is the final resulting cluster tractogram
    """
    MODEL = 'model'
    FILTERED = 'filtered'
    CLEAN = 'clean'

def _download_cluster_tractograms(expirement_name, base_dir, subjects, session_names, bundle_name, range_n_clusters, cluster_type):
    """
    Download all cluster tractograms for the given expirement. 
    
    This is for each bundle for each subject, session, and desired number of clusters.

        `{expirement_name}/{bundle_name}/{subject}/{session}/{n_clusters}/{model_name}_cluster_{cluster_label}_{cluster_type}.trk`

    Parameters
    ----------
    expirement_name : string
    base_dir : string
    subjects : list
    session_names : list
    bundle_names : list
    range_n_clusters : list
    cluster_type : str

    Returns
    -------
    dictionary has `subject` as key containing
    └── dictionary with `session_name` as key containing
        └── dictionary with `n_clusters` as key
            └── list of cluster tractograms
    """
    import s3fs
    from os.path import exists, join, basename
    from dipy.io.streamline import load_tractogram

    fs = s3fs.S3FileSystem()

    cluster_tractograms = {}

    for subject in subjects:
        cluster_tractograms[subject] = {}

        for session in session_names:
            cluster_tractograms[subject][session] = {}

            for n_clusters in range_n_clusters:
                cluster_tractograms[subject][session][n_clusters] = []

                # only download the final cluster tractograms and generate correspoinding density maps
                remote_cluster_tractograms = fs.glob(f'hcp-subbundle/{expirement_name}/{session}/{bundle_name}/{subject}/{n_clusters}/*_{cluster_type}.trk')

                for remote_cluster_tractogram in remote_cluster_tractograms:
                    cluster_tractogram_basename = basename(remote_cluster_tractogram)

                    local_cluster_tractogram = join(base_dir, subject, session, str(n_clusters), cluster_tractogram_basename)
                    
                    if not exists(local_cluster_tractogram):
                        logger.log(logging.DEBUG, f'downloading {local_cluster_tractogram}')
                        fs.get(remote_cluster_tractogram, local_cluster_tractogram)

                    logger.log(logging.DEBUG, f'loading {local_cluster_tractogram}')
                    tractogram = load_tractogram(local_cluster_tractogram, 'same')
                    cluster_tractograms[subject][session][n_clusters].append(tractogram)

    return cluster_tractograms

def _download_bundle_fa_profiles(expirement_name, base_dir, subjects, session_names, bundle_name, range_n_clusters):
    """
    Download the bundle FA profile. 
    
    The bundle profile is the same regardless of n_clusters, therefore can just 
    download first one.

    Only used for quality control purposes.

    Parameters
    ----------
    expirement_name : string
    base_dir : string
    subjects : list
    session_names : list
    bundle_names : list
    range_n_clusters : list

    Returns
    -------
    profiles : dict
        dictionary with `subject` as key containing
        └── dictionary with `session_name` as key
            └── containing the bundle fa profile
    """
    import s3fs
    from os.path import basename, exists, join
    import numpy as np

    fs = s3fs.S3FileSystem()

    profiles = {}

    for subject in subjects:
        profiles[subject] = {}

        for session in session_names:
            profiles[subject][session] = {}

            n_clusters = range_n_clusters[0]
                
            remote_profile = f'hcp-subbundle/{expirement_name}/{session}/{bundle_name}/{subject}/{n_clusters}/bundle_profile_fa.npy'

            local_profile = join(base_dir, subject, session, basename(remote_profile))
            
            if not exists(local_profile):
                logger.log(logging.DEBUG, f'downloading {local_profile}')
                fs.get(remote_profile, local_profile)
            
            profiles[subject][session] = np.load(local_profile)

    return profiles


def _download_streamline_fa_profiles(expirement_name, base_dir, subjects, session_names, bundle_name, range_n_clusters):
    """
    Download the streamline FA profiles.
    
    The streamline profiles are the same regardless of n_clusters, therefore can just 
    download first one.

    Only used for quality control purposes.

    Parameters
    ----------
    expirement_name : string
    base_dir : string
    subjects : list
    session_names : list
    bundle_names : list
    range_n_clusters : list

    Returns
    -------
    profiles : dict
        dictionary with `subject` as key containing
        └── dictionary with `session_name` as key
            └── containing the streamline fa profiles
    """

    import s3fs
    from os.path import basename, exists, join
    import numpy as np

    fs = s3fs.S3FileSystem()

    profiles = {}

    for subject in subjects:
        profiles[subject] = {}

        for session in session_names:
            profiles[subject][session] = {}

            n_clusters = range_n_clusters[0]

            remote_profile = f'hcp-subbundle/{expirement_name}/{session}/{bundle_name}/{subject}/{n_clusters}/streamline_profile_fa.npy'

            local_profile = join(base_dir, subject, session, basename(remote_profile))
                    
            if not exists(local_profile):
                logger.log(logging.DEBUG, f'downloading {local_profile}')
                fs.get(remote_profile, local_profile)
            
            profiles[subject][session] = np.load(local_profile)

    return profiles


def _download_cluster_fa_profiles(expirement_name, base_dir, subjects, session_names, bundle_name, range_n_clusters):
    """
    Download the cluster FA profiles.
    
    Only used for quality control purposes.

    Parameters
    ----------
    expirement_name : string
    base_dir : string
    subjects : list
    session_names : list
    bundle_names : list
    range_n_clusters : list

    Returns
    -------
    profiles : dict
        dictionary with `subject` as key containing
        └── dictionary with `session_name` as key
            └── containing list of cluster fa profiles
    """
    import s3fs
    from os.path import basename, exists, join
    import numpy as np

    fs = s3fs.S3FileSystem()

    profiles = {}

    for subject in subjects:
        profiles[subject] = {}

        for session in session_names:
            profiles[subject][session] = {}

            for n_clusters in range_n_clusters:
                profiles[subject][session][n_clusters] = []

                remote_profiles = fs.glob(f'hcp-subbundle/{expirement_name}/{session}/{bundle_name}/{subject}/{n_clusters}/cluster_*_profile_fa.npy')

                for remote_profile in remote_profiles:
                    local_profile = join(base_dir, subject, session, str(n_clusters), basename(remote_profile))
                    
                    if not exists(local_profile):
                        logger.log(logging.DEBUG, f'downloading {local_profile}')
                        fs.get(remote_profile, local_profile)
                    
                    profiles[subject][session][n_clusters].append(np.load(local_profile))

    return profiles


def fetch_model_data(metadata):
    """
    set up local directory and download necessary files for model analysis

    Parameters
    ----------
    metadata : dict

    Returns
    -------
    model_data : dict
        dictionary with `bundle_name` as key containing
        └── dictionary with following keys:
                'fa_scalar_data', 'md_scalar_data', 
                'adjacencies',
                'embeddings', 'cluster_labels', 'filtered_embeddings', 'filtered_cluster_labels',
                'bundle_tractograms', 
                'model_cluster_tractograms', 'filtered_cluster_tractograms', 'clean_cluster_tractograms',
                'bundle_profiles', 'streamline_profiles', 'cluster_profiles'
    """
    from os import makedirs
    from os.path import join

    model_data = {}

    for bundle_name in metadata['experiment_bundles']:
        model_data[bundle_name] = {}

        base_dir = join(metadata['experiment_output_dir'], bundle_name)

        # ensure local directories exist
        for subject in metadata['experiment_subjects']:
            for session in metadata['experiment_sessions']:
                for cluster_number in metadata['experiment_range_n_clusters']:
                    makedirs(join(base_dir, subject, session, str(cluster_number)), exist_ok=True)
        
        logger.log(logging.INFO, f'Download {bundle_name} data from HCP reliability study')
        
        for scalar in metadata['model_scalars']:
            scalar_abr = scalar.split('.')[0]
            model_data[bundle_name][f'{scalar_abr.lower()}_scalar_data'] = _download_scalar_data(
                scalar, base_dir, metadata['experiment_subjects'], metadata['experiment_sessions']
            )

        model_data[bundle_name]['adjacencies'] = _download_adjacencies(
            metadata['experiment_name'], 
            base_dir, 
            metadata['experiment_subjects'], 
            metadata['experiment_sessions'], 
            bundle_name, 
            metadata['experiment_range_n_clusters']
        )

        model_data[bundle_name]['bundle_tractograms'] = _download_bundle_tractograms(
            base_dir, metadata['experiment_subjects'], metadata['experiment_sessions'], bundle_name
        )

        # originals from model fit
        model_data[bundle_name]['embeddings'] = _download_cluster_artifacts(
            metadata['experiment_name'], 
            base_dir, 
            metadata['experiment_subjects'], 
            metadata['experiment_sessions'], 
            bundle_name, 
            metadata['experiment_range_n_clusters'],
            ArtifactType.EMBEDDINGS
        )
        model_data[bundle_name]['cluster_labels'] = _download_cluster_artifacts(
            metadata['experiment_name'], 
            base_dir, 
            metadata['experiment_subjects'], 
            metadata['experiment_sessions'], 
            bundle_name, 
            metadata['experiment_range_n_clusters'],
            ArtifactType.CLUSTER_LABELS
        )

        # removed streamlines below the average silhouette score
        model_data[bundle_name]['filtered_embeddings'] = _download_cluster_artifacts(
            metadata['experiment_name'], 
            base_dir, 
            metadata['experiment_subjects'], 
            metadata['experiment_sessions'], 
            bundle_name, 
            metadata['experiment_range_n_clusters'],
            ArtifactType.EMBEDDINGS_FILTERED
        )
        model_data[bundle_name]['filtered_cluster_labels'] = _download_cluster_artifacts(
            metadata['experiment_name'], 
            base_dir, 
            metadata['experiment_subjects'], 
            metadata['experiment_sessions'], 
            bundle_name, 
            metadata['experiment_range_n_clusters'],
            ArtifactType.CLUSTER_LABELS_FILTERED
        )

        logger.log(logging.INFO, f"Download {bundle_name} clustering models for K={metadata['experiment_range_n_clusters']}")
        # original from model fit
        model_data[bundle_name]['model_cluster_tractograms'] = _download_cluster_tractograms(
            metadata['experiment_name'], 
            base_dir, 
            metadata['experiment_subjects'], 
            metadata['experiment_sessions'], 
            bundle_name, 
            metadata['experiment_range_n_clusters'],
            ClusterType.MODEL
        )
        # removed streamlines below the average silhouette score
        model_data[bundle_name]['filtered_cluster_tractograms'] = _download_cluster_tractograms(
            metadata['experiment_name'], 
            base_dir, 
            metadata['experiment_subjects'], 
            metadata['experiment_sessions'], 
            bundle_name, 
            metadata['experiment_range_n_clusters'],
            ClusterType.FILTERED
        )
        # cleaned
        model_data[bundle_name]['clean_cluster_tractograms'] = _download_cluster_tractograms(
            metadata['experiment_name'], 
            base_dir, 
            metadata['experiment_subjects'], 
            metadata['experiment_sessions'], 
            bundle_name, 
            metadata['experiment_range_n_clusters'],
            ClusterType.CLEAN
        )

        # profiles
        model_data[bundle_name]['bundle_profiles'] = _download_bundle_fa_profiles(
            metadata['experiment_name'], 
            base_dir, 
            metadata['experiment_subjects'], 
            metadata['experiment_sessions'], 
            bundle_name, 
            metadata['experiment_range_n_clusters']
        )
        model_data[bundle_name]['streamline_profiles'] = _download_streamline_fa_profiles(
            metadata['experiment_name'], 
            base_dir, 
            metadata['experiment_subjects'], 
            metadata['experiment_sessions'], 
            bundle_name, 
            metadata['experiment_range_n_clusters']
        )
        model_data[bundle_name]['cluster_profiles'] = _download_cluster_fa_profiles(
            metadata['experiment_name'], 
            base_dir, 
            metadata['experiment_subjects'], 
            metadata['experiment_sessions'], 
            bundle_name, 
            metadata['experiment_range_n_clusters']
        )

    return model_data

def make_bundle_dict(metadata):
    """
    create a pyAFQ bundle dictionary object for the largest number of clusters
    in the experiment
    """
    bundle_dict = {}
    
    maximal_n_clusters = max(metadata['experiment_range_n_clusters'])
    for bundle_name in metadata['experiment_bundles']:
        # bundle_name_prefix = bundle_name.split('_')[0]
        
        for cluster_id in range(maximal_n_clusters):
            bundle_dict[bundle_name + '_' + str(cluster_id)] = {"uid" : cluster_id}
            # bundle_dict[bundle_name_prefix + '_' + str(cluster_id)] = {"uid" : cluster_id}
        
    return bundle_dict
