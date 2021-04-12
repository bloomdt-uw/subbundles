"""
utility functions to support subbundle model analysis
"""
import logging
logger = logging.getLogger('subbundle')

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

def fetch_model_data(expirement_name, subjects, session_names, bundle_names, cluster_numbers):
    """
    set up local directory and download necessary files for model analysis

    Parameters
    ----------
    expirement_name : string
    subjects : list
    session_names : list
    bundle_names : list
    cluster_numbers : list

    Returns
    -------
    model_data : dict
        dictionary with `bundle_name` as key containing
        └── dictionary with following keys:
                'fa_scalar_data', 'md_scalar_data', 'tractograms'
                'model_names', 'cluster_idxs', 'cluster_names', 
                'cluster_tractograms', 'cluster_tractograms_clean',
                'cluster_denisty_maps', 'cluster_denisty_maps_clean'
    """
    from os import makedirs
    from os.path import join

    model_data = {}

    for bundle_name in bundle_names:
        model_data[bundle_name] = {}

        base_dir = join('subbundles', expirement_name, bundle_name)

        # ensure local directories exist
        for subject in subjects:
            for session in session_names:
                for cluster_number in cluster_numbers:
                    makedirs(join(base_dir, subject, session, str(cluster_number)), exist_ok=True)
        
        logger.log(logging.INFO, f'Download {bundle_name} data from HCP reliability study')
        # NOTE since organized by bundle then by subject this downloads the subjects
        # data multiple times, really only need it once per subject/session and not
        # for each bundle. this is inefficent computationally, bandwidth, and storage,
        # may want to reconsider how best to orgainizes.
        # TODO for moment assuming that model is using both FA and MD tissue properties;
        # in future should inspect model metadata to determine scalars
        model_data[bundle_name]['fa_scalar_data'] = _download_scalar_data('FA', base_dir, subjects, session_names)
        model_data[bundle_name]['md_scalar_data'] = _download_scalar_data('MD', base_dir, subjects, session_names)
        model_data[bundle_name]['tractograms'] = _download_bundle_tractograms(base_dir, subjects, session_names, bundle_name)

        logger.log(logging.INFO, f'Download {bundle_name} clustering models for K={cluster_numbers}')
        (model_names, _, cluster_idxs, cluster_names, _, 
        cluster_tractograms, cluster_tractograms_clean, 
        cluster_denisty_maps, cluster_denisty_maps_clean) = _download_clusters(
            expirement_name, base_dir, subjects, session_names, bundle_name, cluster_numbers
        )
        
        model_data[bundle_name]['model_names'] = model_names
        model_data[bundle_name]['cluster_idxs'] = cluster_idxs
        model_data[bundle_name]['cluster_names'] = cluster_names
        model_data[bundle_name]['cluster_tractograms'] = cluster_tractograms
        model_data[bundle_name]['cluster_tractograms_clean'] = cluster_tractograms_clean
        model_data[bundle_name]['cluster_denisty_maps'] = cluster_denisty_maps
        model_data[bundle_name]['cluster_denisty_maps_clean'] = cluster_denisty_maps_clean

    return model_data


def clean_tractogram(tractogram, tractogram_filename):
    """
    Take a tractogram and run cleaning to remove extraneous streamlines.
    
    Saves the cleaned version to disk.

    Parameters
    ----------
    tractogram : StatefulTractogram
    tractogram_filename : string

    Returns
    -------
    sft : StatefulTractogram
    clean_filename : string
    """
    from AFQ.segmentation import clean_bundle
    from dipy.io.stateful_tractogram import StatefulTractogram
    from os.path import exists, splitext
    from dipy.io.streamline import load_tractogram, save_tractogram
    import logging

    # already using from_sft, suppress warnings from dipy
    logging.getLogger("StatefulTractogram").setLevel(logging.ERROR)

    base, ext = splitext(tractogram_filename)
    clean_filename = base + '_clean' + ext

    if not exists(clean_filename):
        logger.log(logging.DEBUG, f'generating {clean_filename}')

        clean_tractogram = clean_bundle(tractogram)
        sft = StatefulTractogram.from_sft(clean_tractogram.streamlines, tractogram)
        
        logger.log(logging.DEBUG, f'saving {clean_filename}')
        save_tractogram(sft, clean_filename, False)
    else:
        logger.log(logging.DEBUG, f'loading {clean_filename}')
        sft = load_tractogram(clean_filename, 'same')

    return sft, clean_filename

def get_density_map(tractogram, tractogram_filename):
    """
    Take a tractogram and return a binary image of the streamlines,
    these images is used to calculate the dice coefficents to compare
    cluster similiartiy.

    Saves the density map

    Parameters
    ----------
    tractogram : StatefulTractogram
    tractogram_filename : string

    Returns
    -------
    density_map_img : Nifti1Image
    density_map_filename : string
    """
    import numpy as np
    from dipy.io.utils import create_nifti_header, get_reference_info
    from dipy.io.stateful_tractogram import Space
    import dipy.tracking.utils as dtu
    from os.path import exists, splitext
    import nibabel as nib

    base, _ = splitext(tractogram_filename)
    density_map_filename = base + '_density_map.nii.gz'

    if not exists(density_map_filename):
        logger.log(logging.DEBUG, f'generating {density_map_filename}')

        if (tractogram._space != Space.VOX):
            tractogram.to_vox()

        affine, vol_dims, voxel_sizes, voxel_order = get_reference_info(tractogram)
        tractogram_density = dtu.density_map(tractogram.streamlines, np.eye(4), vol_dims)
        # force to unsigned 8-bit; done to reduce the size of the density map image
        tractogram_density = np.uint8(tractogram_density)
        nifti_header = create_nifti_header(affine, vol_dims, voxel_sizes)
        density_map_img = nib.Nifti1Image(tractogram_density, affine, nifti_header)

        logger.log(logging.DEBUG, f'saving {density_map_filename}')
        nib.save(density_map_img, density_map_filename)
    else:
        logger.log(logging.DEBUG, f'loading {density_map_filename}')
        density_map_img = nib.load(density_map_filename)

    return density_map_img, density_map_filename

def _download_scalar_data(scalar_name, base_dir, subjects, session_names, use_csd=True):
    """
    Download scalar data for `scalar_name` for all subjects and sessions from the
    single shell HCP reliability study. By default will download the csd scalar data.

    Parameters
    ----------
    scalar_name : string
        either 'FA' or 'MD'
    base_dir : string
    subjects : array
    session_names : array
    use_csd : boolean
        default True

    Returns
    -------
    scalar_data : dict
        dictionary with `subject` as key containing
        └── dictionary with `session_name` as key
        containing the scalar image fdata for that subject and session.
    """
    import s3fs
    from os.path import exists, join
    import nibabel as nib

    if use_csd:
        csd_suffix = '_CSD'
    else:
        csd_suffix = ''

    fs = s3fs.S3FileSystem()

    scalar_basename = f'{scalar_name}.nii.gz'

    scalar_data = {}

    for subject in subjects:
        scalar_data[subject] = {}
        for session in session_names:
            scalar_filename = join(base_dir, subject, session, scalar_basename)
            if not exists(scalar_filename):
                logger.log(logging.DEBUG, f'downloading {scalar_filename}')
                fs.get(
                    (
                        f'profile-hcp-west/hcp_reliability/single_shell/'
                        f'{session.lower()}_afq{csd_suffix}/'
                        f'sub-{subject}/ses-01/'
                        f'sub-{subject}_dwi_model-DTI_{scalar_name}.nii.gz'
                    ),
                    scalar_filename
                )

            logger.log(logging.DEBUG, f'loading {scalar_filename}')
            scalar_data[subject][session] = nib.load(scalar_filename).get_fdata()

    return scalar_data


def _download_bundle_tractograms(base_dir, subjects, session_names, bundle_name, use_csd=True, use_clean=True, generate_metadata=False):
    """
    Download the bundle tractogram for all subjects and sessions from the single shell
    HCP reliabilty study. By default will download clean csd tractograms.

    Will optionally create `streamline_counts.csv` and `streamline_counts.png` in 
    `base_dir`.

    Parameters
    ----------
    base_dir : string
    subjects : array
    session_names : array
    use_csd : boolean
        default True
    clean : boolean
        default True
    generate_metadata : boolean
        default False

    Returns
    -------
    scalar_data : dict
        dictionary with `subject` as key containing
        └── dictionary with `session_name` as key
        containing the bundle `StatefulTractogram` for that subject and session.
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
    
    if generate_metadata:
        streamline_counts = {}

    for subject in subjects:
        tractograms[subject] = {}
        if generate_metadata:
            streamline_counts[subject] = {}

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

            if generate_metadata:
                streamline_counts[subject][session] = len(tractogram.streamlines)


    if generate_metadata:
        import pandas as pd
        import matplotlib.pyplot as plt

        pd.DataFrame(streamline_counts).to_csv(join(base_dir, 'streamline_counts.csv'))
        pd.DataFrame(streamline_counts).T[1:].plot(kind='bar')
        plt.savefig(join(base_dir, 'streamline_counts.png'))
        plt.close()

    return tractograms


def _download_clusters(expirement_name, base_dir, subjects, session_names, bundle_name, cluster_numbers, generate_metadata=False):
    """
    Download all cluster assignment files for the given expirement. This is for each bundle for each subject, 
    session, and desired number of clusters.

    optionally generates three csv files: `model_names.csv`, `cluster_names.csv`, and `cluster_counts.csv`
    in `base_dir`

    Parameters
    ----------
    expirement_name : string
    base_dir : string
    subjects : list
    session_names : list
    bundle_names : list
    cluster_numbers : list
    generate_metadata : boolean
        default False

    Returns
    -------
    Muliple mutlilevel dictionaries. 
    
    Each dictionary has `subject` as key containing
    └── dictionary with `session_name` as key containing
        └── dictionary with `cluster_number` as key

    model_names : dict
        containing the name of the model. by convention includes abbreviation for clustering algorithm and adjacencies.

        for example:
            ['mase_kmeans_fa_r2_md_r2_is_mdf']

        useful if there are mutliple models, to be able to align values in other dictionaries

    cluster_labels : dict
        an array of clusters assigned by the model to each streamline as index. this is the only artifact from clustering. 
        all other dictionaries are derived from this information.

        for example: 
            [0, 0, 1, 3, 3, 2, 1 .... ]
    
    cluster_idxs : dict
        an array for each cluster, listing the corresponding streamline indexes

        for example: 
            [[0, 1, ...], [2, 6, ...], [5, ...], [3, 4, ...], ...]

    cluster_names : dict
        an array with the cluster name. necessary as some models only maximally partition data into K=`cluster_number`.

        for example: 
            if `K=3` the model may only return 2 clusters with names [0, 1].

    cluster_counts : dict
        an array with the number of streamlines assigned to each cluster.

    cluster_tractograms : dict
    cluster_tractograms_clean : dict
    cluster_density_maps : dict
    cluster_density_maps_clean : dict
    """
    import s3fs
    from os.path import exists, join, basename, splitext
    import numpy as np
    from dipy.io.streamline import load_tractogram
    import nibabel as nib

    fs = s3fs.S3FileSystem()

    model_names = {}
    cluster_labels = {}
    cluster_idxs = {}
    cluster_names = {}
    cluster_counts = {}
    cluster_tractograms = {}
    cluster_tractograms_clean = {}
    cluster_density_maps = {}
    cluster_density_maps_clean = {}

    for subject in subjects:
        model_names[subject] = {}
        cluster_labels[subject] = {}
        cluster_idxs[subject] = {}
        cluster_names[subject] = {}
        cluster_counts[subject] = {}
        cluster_tractograms[subject] = {}
        cluster_tractograms_clean[subject] = {}
        cluster_density_maps[subject] = {}
        cluster_density_maps_clean[subject] = {}

        for session in session_names:
            model_names[subject][session] = {}
            cluster_labels[subject][session] = {}
            cluster_idxs[subject][session] = {}
            cluster_names[subject][session] = {}
            cluster_counts[subject][session] = {}
            cluster_tractograms[subject][session] = {}
            cluster_tractograms_clean[subject][session] = {}
            cluster_density_maps[subject][session] = {}
            cluster_density_maps_clean[subject][session] = {}

            for cluster_number in cluster_numbers:
                model_names[subject][session][cluster_number] = []
                cluster_labels[subject][session][cluster_number] = []
                cluster_idxs[subject][session][cluster_number] = []
                cluster_names[subject][session][cluster_number] = []
                cluster_counts[subject][session][cluster_number] = []
                cluster_tractograms[subject][session][cluster_number] = []
                cluster_tractograms_clean[subject][session][cluster_number] = []
                cluster_density_maps[subject][session][cluster_number] = []
                cluster_density_maps_clean[subject][session][cluster_number] = []

                # At this point typically only one model per cluster,
                # but it is possibile for multiple models:
                # e.g. considering effects of different tissue properties or
                # clustering algorithms
                # NOTE model are sorted alphabetically
                remote_cluster_filenames = fs.glob(f'hcp-subbundle/{expirement_name}/{session}/{bundle_name}/{subject}/{cluster_number}/*idx.npy')

                for remote_cluter_filename in remote_cluster_filenames:
                    # print(subject, session, remote_cluter_filename)
                    cluster_basename = basename(remote_cluter_filename)
                    local_cluster_filename = join(base_dir, subject, session, str(cluster_number), cluster_basename)

                    if not exists(local_cluster_filename):
                        logger.log(logging.DEBUG, f'downloading {local_cluster_filename}')
                        fs.get(remote_cluter_filename, local_cluster_filename)

                    cluster_rootname, _ = splitext(cluster_basename)
                    cluster_rootname = cluster_rootname.rsplit('_',1)[0]

                    logger.log(logging.DEBUG, f'deriving {cluster_rootname} metadata')
                    model_names[subject][session][cluster_number].append(cluster_rootname)

                    logger.log(logging.DEBUG, f'loading {local_cluster_filename}')
                    sorted_cluster_labels = np.load(local_cluster_filename)
                    cluster_labels[subject][session][cluster_number].append(sorted_cluster_labels)

                    cluster_names[subject][session][cluster_number].append(np.unique(sorted_cluster_labels))
                    cluster_idxs[subject][session][cluster_number].append(np.array([np.where(sorted_cluster_labels == i)[0] for i in np.unique(sorted_cluster_labels)]))
                    cluster_counts[subject][session][cluster_number].append(np.bincount(sorted_cluster_labels))

                # download the cluster tractograms and density maps
                remote_cluster_tractograms = fs.glob(f'hcp-subbundle/{expirement_name}/{session}/{bundle_name}/{subject}/{cluster_number}/*.trk')

                for remote_cluster_tractogram in remote_cluster_tractograms:
                    
                    # do not redownload the bundle tractogram
                    if (remote_cluster_tractogram.endswith(f'{bundle_name}.trk')):
                        continue

                    # tractogram
                    cluster_tractogram_basename = basename(remote_cluster_tractogram)
                    cluster_rootname, _ = splitext(cluster_tractogram_basename)

                    local_cluster_tractogram = join(base_dir, subject, session, str(cluster_number), cluster_tractogram_basename)
                    
                    if not exists(local_cluster_tractogram):
                        logger.log(logging.DEBUG, f'downloading {local_cluster_tractogram}')
                        fs.get(remote_cluster_tractogram, local_cluster_tractogram)

                    logger.log(logging.DEBUG, f'loading {local_cluster_tractogram}')
                    tractogram = load_tractogram(local_cluster_tractogram, 'same')
                    cluster_tractograms[subject][session][cluster_number].append(tractogram)
                    
                    # clean the cluster tractogram
                    cleaned_tractogram, cleaned_tractogram_filename = clean_tractogram(tractogram, local_cluster_tractogram)
                    cluster_tractograms_clean[subject][session][cluster_number].append(cleaned_tractogram)
                    
                    # density map
                    cluster_density_map_basename = cluster_rootname + '_density_map.nii.gz'
                    remote_cluster_density_map = f'hcp-subbundle/{expirement_name}/{session}/{bundle_name}/{subject}/{cluster_number}/{cluster_density_map_basename}'
                    local_cluster_density_map = join(base_dir, subject, session, str(cluster_number), cluster_density_map_basename)
                    
                    if not exists(local_cluster_density_map) and fs.exists(remote_cluster_density_map):
                        logger.log(logging.DEBUG, f'downloading {local_cluster_density_map}')
                        fs.get(remote_cluster_density_map, local_cluster_density_map)

                    # if density map doesn't exist on S3 generate locally
                    if not exists(local_cluster_density_map):
                        logger.log(logging.DEBUG, f'generating {local_cluster_density_map}')
                        get_density_map(tractogram, local_cluster_tractogram)

                    cluster_density_maps[subject][session][cluster_number].append(nib.load(local_cluster_density_map))

                    # density map from clean tractogram
                    cleaned_denisty_map, _ = get_density_map(cleaned_tractogram, cleaned_tractogram_filename)
                    cluster_density_maps_clean[subject][session][cluster_number].append(cleaned_denisty_map)


    if generate_metadata:
        import pandas as pd
        pd.DataFrame(model_names).to_csv(join(base_dir, 'model_names.csv'))
        pd.DataFrame(cluster_names).to_csv(join(base_dir, 'cluster_names.csv'))
        pd.DataFrame(cluster_counts).to_csv(join(base_dir, 'cluster_counts.csv'))

    return (
        model_names, 
        cluster_labels, cluster_idxs, cluster_names, cluster_counts, 
        cluster_tractograms, cluster_tractograms_clean,
        cluster_density_maps, cluster_density_maps_clean
    )