"""
Data driven approach to identify subbundles across subjects from HCP_1200
(the HCP dataset test session)
• Move each subjects cluster into MNI space
• Calculate weighted dice coefficient to identify cluster similarity across
  subjects. For each subject pair results in NxM matrix where N is number of
  clusters in first subject and M is number of clusters in second subject.
• Find the optimal cluster alignment for given subject pair using maximal
  trace on the NxM matrix. This approach will preserve at least min(N,M) 
  clusters.
• Choose a target subject and find relabeling for all other subjects in 
  dataset. Then calculate the cluster profiles with new label. Iterate
  through all subjects as the target subject. This 'Leave one out' approach
  should identify the 'consensus subject' for dataset -- which is the subject
  with minimum variance across cluster profiles.
• Using same consensus subject recalculate labels for HCP_retest (the HCP
  dataset retest session).
• Then conduct test-retest relability analysis as in visualizations.py
"""
import logging
logger = logging.getLogger('subbundle')

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


class Algorithm:
    """
    enum representing supported aglorithms to find matching clustering labels
    across models
    """
    MAXDICE = 'maxdice'
    MUNKRES = 'munkres'
    CENTROID = 'mdf'

algorithms = [Algorithm.MAXDICE, Algorithm.MUNKRES, Algorithm.CENTROID]

###############################################################################
###############################################################################
# Helper function
###############################################################################
###############################################################################

def get_cluster_info(metadata, consensus_subjects=None):
    """
    Construct cluster centroids and tractograms in MNI space from test session
    so can run `match_clusters`

    Identify `consensus_subject` for the test session n_clusters using `metadata['algorithm']`.
    The consensus subject is use to align clusters across test-retest.

    Parameters
    ----------
    metadata : dict

    Returns
    -------
    cluster_info : dict `n_clusters` key
    └── values dict with following keys : 
        `consensus_subject` and `session_name` as keys
        └── `consensus_subject` : `subject_id`
        └── `session_name` : dict with centriods`, `tractograms_filenames`, `tractograms` keys
            └── `centriods` : dict with `subject_id` key
                └── values centroid `StatefulTratogram` list of length `n_clusters`
            └── `tractograms_filenames` : dict with `subject_id` key
                └── tractogram filename list of length `n_clusters`
            └── `tractograms` : dict with `subject_id` key
                └── `StatefulTractogram` list of length n_clusters
    """
    from os.path import join

    cluster_info = {}

    base_dir = join(metadata['experiment_output_dir'], metadata['bundle_name'])

    for n_clusters in metadata['experiment_range_n_clusters']:
        cluster_info[n_clusters] = {}

        for session_name in metadata['experiment_sessions']:
            cluster_info[n_clusters][session_name] = {}

            # (test session) centroids from cleaned cluster tractograms
            # for each subject and cluster_label find the cluster centroid
            prealign_centroids = _prealignment_centroids(
                base_dir,
                session_name,
                metadata['model_name'],
                metadata['experiment_subjects'],
                metadata['bundle_name'],
                n_clusters
            )
            
            # (test session) MNI centroids from cleaned cluster tractograms
            # move the centroids into MNI space
            #
            # used for:
            # • visualization and 
            # • matching test clusters with Algorithm.CENTROID
            cluster_info[n_clusters][session_name]['centroids'] = _move_centroids_to_MNI(
                session_name,
                metadata['experiment_subjects'],
                prealign_centroids
            )
            
            # (test session) MNI tractograms
            # for each subject move all cleaned cluster tractograms into the MNI space
            # allows for easier comparision using weighted dice
            # 
            # used for:
            # • matching test clusters with using Algorithm.MAXDICE or Algorithm.MUNKRES
            tractogram_dict, tractograms_filename_dicts = _load_MNI_cluster_tractograms(
                base_dir,
                session_name,
                metadata['model_name'],
                metadata['experiment_subjects'],
                metadata['bundle_name'],
                n_clusters
            )
            
            cluster_info[n_clusters][session_name]['tractograms_filenames'] = tractograms_filename_dicts
            cluster_info[n_clusters][session_name]['tractograms'] = tractogram_dict

        # consensus_subject
        # once we have the centroids and tractograms we can calculate the consensus subject
        # using the specified algorithm
        #
        # this could be extracted into a separate step:
        # then could determine if consensus subjects are consistent across algorithms.
        # right now just run one algorithm at a time.
        if consensus_subjects is None:
            cluster_info[n_clusters]['consensus_subject'] = _find_consensus_subject(
                base_dir,
                metadata['experiment_test_session'], 
                metadata['model_name'],
                metadata['experiment_subjects'],
                cluster_info,
                metadata['bundle_name'],
                n_clusters,
                metadata['algorithm']
            )
        else:
            cluster_info[n_clusters]['consensus_subject'] = consensus_subjects[n_clusters]['consensus_subject']

    return cluster_info


def get_relabeled_centroids(metadata, n_clusters, session_name, consensus):
    """
    Used in visualizations.py
    """
    from os.path import join

    base_dir = join(metadata['experiment_output_dir'], metadata['bundle_name'])

    cluster_labels = _get_relabeled_clusters(
        base_dir,
        session_name,
        metadata['experiment_subjects'],
        n_clusters,
        consensus,
        metadata['algorithm']
    )

    relabeled_centroids = _relabeled_centroids(
        base_dir,
        session_name,
        metadata['model_name'],
        metadata['experiment_subjects'],
        metadata['bundle_name'],
        n_clusters, 
        cluster_labels
    )

    return _move_centroids_to_MNI(
        session_name,
        metadata['experiment_subjects'],
        relabeled_centroids
    )

###############################################################################
###############################################################################
# Step 0:
# • Move clusters into MNI space
###############################################################################
###############################################################################

def move_tractogram_to_MNI_space(session_name, subject, tractogram):
    """
    For given subject and tractogram move that tractogram from subject space
    to MNI space using existing AFQ derivatives.

    Looks for:
    - subjects DWI file
        └── `~/AFQ_data/{session_name}/deriavatives/dmriprep/sub-{subject}/ses-01/dwi/sub-{subject}_dwi.nii.gz`
        
        if does not exist will attempt to download from HCP repository
    - MNI image
        └── `~/.cache/templateflow/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-01_T2w.nii.gz`
    - mapping DWI to MNI mapping file:
        └── `~/AFQ_data/{session_name}/derivatives/afq/sub-{subject}/ses-01/sub-{subject}_dwi_mapping_from-DWI_to_MNI_xfm.nii.gz`
        
        if does not exist will attempt to download from AFQ HCP repository
    - prealign file:
        └── `~/AFQ_data/{session_name}/derivatives/afq/sub-{subject}/ses-01/sub-{subject}_dwi_prealign_from-DWI_to-MNI_xfm.npy`

        fif does not exist will attempt to download from AFQ HCP repository

    Parameters
    ----------
    session_name : str
        HCP dataset identifier. either 'HCP_1200' or 'HCP_Retest'

    subject : str
        subject identifier for the HCP dataset

    tractogram : StatefuleTractogram
        subjects tractogram object in subject space

    Returns
    -------
    sft : StatefuleTractogram
        subjects tractogram object in MNI space
    """
    from os import makedirs
    from os.path import join, expanduser, exists
    import s3fs
    import numpy as np
    import nibabel as nib
    import AFQ.data as afd
    import AFQ.registration as reg
    import dipy.tracking.streamline as dts
    import dipy.tracking.utils as dtu
    from dipy.io.stateful_tractogram import StatefulTractogram
    from dipy.io.stateful_tractogram import Space

    ###############################################################################
    afq_base_dir = join(expanduser('~'),'AFQ_data')
    afq_derivatives_base_dir = join(afq_base_dir, session_name, 'derivatives')

    ###############################################################################
    subject_path = join(f'sub-{subject}', 'ses-01')

    ###############################################################################
    # load subject dwi image

    dwi_derivatives_dir = join(afq_derivatives_base_dir, 'dmriprep')
    dwi_file = join(dwi_derivatives_dir, subject_path, 'dwi', f'sub-{subject}_dwi.nii.gz')

    # download HCP data for subject
    if not exists(dwi_file):
        logger.log(logging.DEBUG, f'downloading dwi {dwi_file}')
        afd.fetch_hcp([subject], study=session_name)

    # ~/AFQ_data/HCP_1200/derivatives/dmriprep/sub-125525/ses-01/dwi/sub-125525_dwi.nii.gz
    logger.log(logging.DEBUG, f'loading dwi {dwi_file}')
    dwi_img = nib.load(dwi_file)

    ## validate dwi image
    # dwi_img.shape -- (145, 174, 145, 288)
    # dwi_img.affine
    # array([[  -1.25,    0.  ,    0.  ,   90.  ],
    #        [   0.  ,    1.25,    0.  , -126.  ],
    #        [   0.  ,    0.  ,    1.25,  -72.  ],
    #        [   0.  ,    0.  ,    0.  ,    1.  ]])

    ###############################################################################
    # load subject to MNI image and mapping from AFQ

    # ~/.cache/templateflow/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-01_T2w.nii.gz
    logger.log(logging.DEBUG, 'loading MNI template')
    MNI_T2_img = afd.read_mni_template()

    ## validate MNI image
    # MNI_T2_img.shape --  (193, 229, 193)
    # MNI_T2_img.affine
    # array([[   1.,    0.,    0.,  -96.],
    #        [   0.,    1.,    0., -132.],
    #        [   0.,    0.,    1.,  -78.],
    #        [   0.,    0.,    0.,    1.]])

    # NOTE: assuming CSD
    fs = s3fs.S3FileSystem()

    afq_derivatives_dir = join(afq_derivatives_base_dir, 'afq')

    # ~/AFQ_data/HCP_1200/derivatives/afq/sub-125525/ses-01/sub-125525_dwi_mapping_from-DWI_to_MNI_xfm.nii.gz
    mapping_file = join(afq_derivatives_dir, subject_path, f'sub-{subject}_dwi_mapping_from-DWI_to_MNI_xfm.nii.gz')

    if not exists(mapping_file):
        logger.log(logging.DEBUG, f'downloading mapping {mapping_file}')
        makedirs(join(afq_derivatives_dir, subject_path), exist_ok=True)
        fs.get(
            (
                f'profile-hcp-west/hcp_reliability/single_shell/'
                f'{session_name.lower()}_afq_CSD/sub-{subject}/ses-01/'
                f'sub-{subject}_dwi_mapping_from-DWI_to_MNI_xfm.nii.gz'
            ),
            f'{mapping_file}'
        )

    # ~/AFQ_data/HCP_1200/derivatives/afq/sub-125525/ses-01/sub-125525_dwi_prealign_from-DWI_to-MNI_xfm.npy
    prealign_file = join(afq_derivatives_dir, subject_path, f'sub-{subject}_dwi_prealign_from-DWI_to-MNI_xfm.npy')

    if not exists(prealign_file):
        logger.log(logging.DEBUG, f'downloading prealign {prealign_file}')
        makedirs(join(afq_derivatives_dir, subject_path), exist_ok=True)
        fs.get(
            (
                f'profile-hcp-west/hcp_reliability/single_shell/'
                f'{session_name.lower()}_afq_CSD/sub-{subject}/ses-01/'
                f'sub-{subject}_dwi_prealign_from-DWI_to-MNI_xfm.npy'
            ),
            f'{prealign_file}'
        )

    ## validate prealign
    # np.round(np.load(prealign_file))
    # array([[ 1.,  0., -0.,  1.],
    #        [-0.,  1., -0., -0.],
    #        [ 0., -0.,  1., -2.],
    #        [ 0.,  0.,  0.,  1.]])

    # mapping from dwi image to MNI image
    logger.log(logging.DEBUG, f'loading mapping {mapping_file}')
    mapping = reg.read_mapping(mapping_file, dwi_img, MNI_T2_img) # both the forward and backward transformations in MNI space

    ## validate mapping
    # np.shape(mapping.forward) -- (193, 229, 193, 3)
    # np.shape(mapping.backward) -- (193, 229, 193, 3)

    ###############################################################################
    # move subject streamlines into MNI space
    logger.log(logging.DEBUG, 'generating MNI tractogram')
    tractogram.to_vox() # to rasmm to vox for values

    # order of transforms is important
    sl_xform = tractogram.streamlines
    sl_xform = dtu.transform_tracking_output(sl_xform, dwi_img.affine) # voxel to mm scaner
    sl_xform = dtu.transform_tracking_output(sl_xform, np.linalg.inv(MNI_T2_img.affine)) # mm scanner to voxel in MNI
    sl_xform = list(dtu.transform_tracking_output(sl_xform, np.linalg.inv(np.load(prealign_file)))) # apply prealignment in MNI

    delta = dts.values_from_volume(mapping.forward, sl_xform, np.eye(4)) # assume in same coordinate space

    ## validiate delta 
    # check delta values exist (needed to convert to VOX)
    # np.count_nonzero([np.count_nonzero(delta[i]) for i in range(len(delta))])

    moved_sl = [d + s for d, s in zip(delta, sl_xform)]

    sft = StatefulTractogram(moved_sl, MNI_T2_img, Space.VOX)

    return sft


def _load_MNI_cluster_tractograms(base_dir, session_name, model_name, subjects, bundle_name, n_clusters):
    """
    Ensure all cluster tractograms are in MNI for all subjects.

    Once subjects are in MNI space we can compare their clusters to determine
    similarity.
    
    Caches (saves) the resulting MNI tractogram to disk. Therefore, remove these files when using new model.
    └── `{base_dir}/{subject}/{session_name}/{n_clusters}/{subject}_{bundle_name}_{cluster_id}_MNI.trk`

    Looks for clean tractography file for each cluster. The `cluster_id`
    is based on the label assigned by the clustering model for that subject.
    └── `{base_dir}/{subject}/{session_name}/{n_clusters}/{model_name}_cluster_{cluster_id}_clean.trk`

    Parameters
    ----------
    base_dir : str
        base directory for the experiment.
    
    session_name : str
        HCP dataset identifier. either 'HCP_1200' or 'HCP_Retest'

    model_name : str
        clustering model name. used to determine tractography files.
    
    subjects : list
        array of strings representing subject identifiers in the data set. can
        be a subset of subjects.
    
    bundle_name : str
        name of the bundle. correspond to pyAFQ bundle names

    n_clusters : int
        number of clusters

    Returns
    -------
    tractograms : dict
        dict of list of StatefulTractograms. each StatefulTractorgram represents
        the subjects cluster in MNI space. the list contains a StatefulTractogram
        for each cluster assigned by the clustering model. the dict stores each
        list by the subject.
    
    tractograms_filenames : dict
    """
    from os.path import join, exists
    from dipy.io.streamline import load_tractogram, save_tractogram

    tractograms = {}
    tractograms_filenames = {}

    for subject in subjects:
        tractograms[subject] = []
        tractograms_filenames[subject] = []

        subbundle_base_dir = join(base_dir, subject, session_name, str(n_clusters))
        
        for cluster_id in range(n_clusters):
            MNI_tractogram_file = join(subbundle_base_dir, f'{subject}_{bundle_name}_{cluster_id}_MNI.trk')

            # load subject clusters into MNI space
            if not exists(MNI_tractogram_file):
                logger.log(logging.DEBUG, f'generating {MNI_tractogram_file}')
                tractogram_name = f'{model_name}_cluster_{cluster_id}_clean.trk'
                tractogram_file = join(subbundle_base_dir, tractogram_name)
                
                if not exists(tractogram_file):
                    logger.log(logging.INFO, f'{subject} {bundle_name} {cluster_id} not found')
                    continue

                tractogram = load_tractogram(tractogram_file, 'same')

                sft = move_tractogram_to_MNI_space(session_name, subject, tractogram)

                logger.log(logging.DEBUG, f'saving {MNI_tractogram_file}')
                save_tractogram(sft, MNI_tractogram_file, bbox_valid_check=False)
            else:
                logger.log(logging.DEBUG, f'loading {MNI_tractogram_file}')
                sft = load_tractogram(MNI_tractogram_file, 'same', bbox_valid_check=False)
                sft.remove_invalid_streamlines()

            tractograms[subject].append(sft)
            tractograms_filenames[subject].append(MNI_tractogram_file)
    
    return tractograms, tractograms_filenames


###############################################################################
###############################################################################
# Step 1:
# • relabel clusters assuming a single fixed target subject
###############################################################################
###############################################################################

def _get_density_map_img(tractogram):
    import numpy as np
    from dipy.io.utils import create_nifti_header, get_reference_info
    from dipy.io.stateful_tractogram import Space
    import dipy.tracking.utils as dtu
    import nibabel as nib
    
    if (tractogram._space != Space.VOX):
        tractogram.to_vox()

    affine, vol_dims, voxel_sizes, voxel_order = get_reference_info(tractogram)
    tractogram_density = dtu.density_map(tractogram.streamlines, np.eye(4), vol_dims)
    # force to unsigned 8-bit; done to reduce the size of the density map image
    tractogram_density = np.uint8(tractogram_density)
    nifti_header = create_nifti_header(affine, vol_dims, voxel_sizes)
    density_map_img = nib.Nifti1Image(tractogram_density, affine, nifti_header)

    return density_map_img

def _get_density_map(tractogram, tractogram_filename):
    """
    Take a tractogram and return a binary image of the streamlines,
    these images is used to calculate the dice coefficents to compare
    cluster similiartiy.

    Saves the density map

    Parameters
    ----------
    tractogram : StatefulTractogram
    tractogram_filename : str

    Returns
    -------
    density_map_img : Nifti1Image
    density_map_filename : str
    """
    
    from os.path import exists, splitext
    import nibabel as nib

    base, _ = splitext(tractogram_filename)
    density_map_filename = base + '_density_map.nii.gz'

    if not exists(density_map_filename):
        logger.log(logging.DEBUG, f'generating {density_map_filename}')

        density_map_img = _get_density_map_img(tractogram)

        logger.log(logging.DEBUG, f'saving {density_map_filename}')
        nib.save(density_map_img, density_map_filename)
    else:
        logger.log(logging.DEBUG, f'loading {density_map_filename}')
        density_map_img = nib.load(density_map_filename)

    return density_map_img, density_map_filename


def _get_dice_coefficients(source_sft, source_filename, target_sft, target_filename):
    """
    Given two tractograms determine the similary by calculating the weighted
    dice coefficient. Assumes source and target are colocated in same space.

    Parameters
    ----------
    source_sft : StatefulTractogram
    source_filename : str
    target_sft : StatefulTractogram
    target_filename : str

    Returns
    -------
    dice_coeff : int
        weighted dice coefficient
    """
    from AFQ.utils.volume import dice_coeff

    source_sft.to_vox()
    source_map, _ = _get_density_map(source_sft, source_filename)

    target_sft.to_vox()
    target_map, _ = _get_density_map(target_sft, target_filename)

    logger.log(logging.DEBUG, 'calculating dice')
    return dice_coeff(source_map, target_map)


def _relabel_clusters(cluster_labels, new_cluster_labels):
    """
    realign clusters across test-retest using consensus subject

    Parameters
    ----------
    cluster_labels: list
    new_cluster_labels: list

    Returns
    -------
    realigned_cluster_labels: list
    """
    import numpy as np

    label_map = dict(zip(np.arange(len(new_cluster_labels)), new_cluster_labels))

    realigned_cluster_labels = np.copy(cluster_labels)

    for original_label, new_label in label_map.items():
        realigned_cluster_labels[cluster_labels == original_label] = new_label

    return realigned_cluster_labels


def _match_subject_clusters_by_maximum_dice(source, source_tractograms, source_tractograms_filenames, target, target_tractograms, target_tractograms_filenames):
    import numpy as np
    import itertools

    pairwise_dice = np.array([])
    
    # compare source and target clusters using weighted dice
    for ((source_tractogram, target_tractogram), (source_filename, target_filename)) in zip(
        list(itertools.product(source_tractograms[source], target_tractograms[target])), 
        list(itertools.product(source_tractograms_filenames[source], target_tractograms_filenames[target]))):
        pairwise_dice = np.append(pairwise_dice, _get_dice_coefficients(source_tractogram, source_filename, target_tractogram, target_filename))

    pairwise_shape = (len(source_tractograms[source]), len(target_tractograms[target]))
    block = pairwise_dice.reshape(pairwise_shape)
    
    # take the source and move label to correspond to target cluster with highest overlap
    # relabel clusters: i -> ids[i]
    # [1 0 0] => 0 -> 1, 1 -> 0, 2 -> 0 # subject 1 to subject 0
    # [0 0 0] => 0 -> 0, 1 -> 0, 2 -> 0 # subject 2 to subject 0
    # [1 2 0] => 1 -> 0, 1 -> 2, 2 -> 0 # subject 2 to subject 1
    target_labels = np.argmax(block, axis=1)

    logger.log(logging.DEBUG, f'maximum dice {source} {target}\n{block}\n{target_labels}')
    
    return target_labels

def _match_clusters_by_maximum_dice(base_dir, session_name, tractograms, tractograms_filenames, target, sources, n_clusters):
    """
    Once every subjects clusters are located in MNI space, pairwise compare
    the weighted dice coefficents of each cluster. 
    
    This version of the algorithm maps each source cluster to the best target
    cluster that is the one with maximum dice coefficient. 
    
    WARNING: This means that the number of clusters is not preserved, and may 
    collapse multiple clusters identified by the model into a single cluster.
    
    For example three source clusters might all map to a single target cluster.
    """
    from os.path import join
    import numpy as np

    for source in sources:
        target_labels = _match_subject_clusters_by_maximum_dice(
            source,
            tractograms,
            tractograms_filenames,
            target,
            tractograms,
            tractograms_filenames
        )

        # align test-retest
        if session_name == "HCP_Retest":
            consensus_base_dir = join(base_dir, target, session_name, str(n_clusters))
            retest_labels = np.load(join(consensus_base_dir, f'consensus_maxdice_labels.npy'))
            target_labels = _relabel_clusters(target_labels, retest_labels)

        source_subbundle_base_dir = join(base_dir, source, session_name, str(n_clusters))
        label_file = join(source_subbundle_base_dir, f'{target}_maxdice_labels.npy')
        np.save(label_file, target_labels)

###############################################################################
# Find permutation of matrix that maximizes its trace using the Munkres algorithm.
# Source: https://gist.github.com/lebedov/9fa8b5a02a0e764cd40c
# Reference: https://stat.ethz.ch/pipermail/r-help/2010-April/236664.html
###############################################################################
def _maximize_trace(a):
    """
    Returns the cluster labels that maximize the trace of matrix `a`
    """
    import itertools
    import numpy as np
    import munkres

    # there are more clusters in the source subject than in the target subject
    transpose = a.shape[0] > a.shape[1]

    # munkres needs rectangluar maxtrix where number rows are less
    # than or equal to number of columns
    if transpose:
        a = np.transpose(a)
    
    d = np.zeros_like(a)
    b = np.eye(a.shape[0], a.shape[1], dtype=float)
    n = min(a.shape)
    for i, j in itertools.product(range(n), range(n)):
        d[j, i] = sum((b[j, :]-a[i, :])**2)
    m = munkres.Munkres()
    inds = m.compute(d)

    if transpose:
        # find the missing source labels and assign to a target label
        labels = []
        source_labels = [x[1] for x in inds]
        for source_label in range(a.shape[0]+1):
            if source_label in source_labels:
                # find the munkres match
                label = inds[source_labels.index(source_label)][0]
                labels.append(label)
            else:
                # munkres didn't find match for this cluster so fall back to
                # taking maximum dice across the column
                label = np.where(a[:,source_label] == np.max(a[:,source_label]))[0][0]
                labels.append(label)
        return labels
    else:
        return [x[1] for x in inds]


def _match_subject_clusters_by_munkres(source, source_tractograms, source_tractograms_filenames, target, target_tractograms, target_tractograms_filenames):
    import numpy as np
    import itertools

    pairwise_dice = np.array([])

    # compare source and target clusters using weighted dice
    for ((source_tractogram, target_tractogram), (source_filename, target_filename)) in zip(
        list(itertools.product(source_tractograms[source], target_tractograms[target])), 
        list(itertools.product(source_tractograms_filenames[source], target_tractograms_filenames[target]))):
        pairwise_dice = np.append(pairwise_dice, _get_dice_coefficients(source_tractogram, source_filename, target_tractogram, target_filename))

    pairwise_shape = (len(source_tractograms[source]), len(target_tractograms[target]))
    block = pairwise_dice.reshape(pairwise_shape)

    target_labels = _maximize_trace(block)

    logger.log(logging.DEBUG, f'munkres {source} {target}\n{block}\n{target_labels}')

    return target_labels

def _match_clusters_by_munkres(base_dir, session_name, tractograms, tractograms_filenames, target, sources, n_clusters):
    """
    Once every subjects clusters are located in MNI space, pairwise compare
    the wieghted dice coefficents of each cluster.
    """
    from os.path import join
    import numpy as np

    # use remaining subjects for pairwise comparison
    for source in sources:
        target_labels = _match_subject_clusters_by_munkres(
            source,
            tractograms,
            tractograms_filenames,
            target,
            tractograms,
            tractograms_filenames
        )

        # align test-retest
        if session_name == "HCP_Retest":
            consensus_base_dir = join(base_dir, target, session_name, str(n_clusters))
            retest_labels = np.load(join(consensus_base_dir, f'consensus_munkres_labels.npy'))
            target_labels = _relabel_clusters(target_labels, retest_labels)

        source_subbundle_base_dir = join(base_dir, source, session_name, str(n_clusters))
        label_file = join(source_subbundle_base_dir, f'{target}_munkres_labels.npy')
        np.save(label_file, target_labels)

###############################################################################
# Since relabeling using weighted DICE coefficients is underperforming, 
# try using centroid MDF
###############################################################################

def convert_centroids(n_clusters, session_name, mni_centroids, bundle_dict, save_sfts=False):
    """
    create tractogram for each mni cluster containing all subjects
    
    Parameters
    ----------
    n_clusters : int
    mni_centroids : dict
    bundle_dict : dict
    save_sfts : bool
    """
    
    from dipy.io.stateful_tractogram import StatefulTractogram
    from AFQ.utils.streamlines import bundles_to_tgram
    from dipy.io.streamline import save_tractogram
    from AFQ.data import read_mni_template

    clusters = [[] for _ in range(n_clusters)]

    for subject in mni_centroids.keys():
        cluster_id = 0
        for cluster_centroid in mni_centroids[subject]:
            clusters[cluster_id].append(cluster_centroid.streamlines[0])
            cluster_id += 1

    # any subject/tractogram will do, so just grab first one
    subject = next(iter(mni_centroids))
    tractogram = mni_centroids[subject][0]
    
    bundles = {}
    
    cluster_id = 0
    for bundle_name in bundle_dict.keys():
        cluster_centroid_sft = StatefulTractogram.from_sft(clusters[cluster_id], tractogram)
        
        if save_sfts == True:
            # temporary saving for ariel
            save_tractogram(cluster_centroid_sft, f'output/{bundle_name}_n_clusters_{n_clusters}_{session_name}_centroid.trk', bbox_valid_check=False)

        bundles[bundle_name] = cluster_centroid_sft
        cluster_id += 1

        # note if bundle dict contains more cluster definitions then n_clusters we cans stop
        if cluster_id == n_clusters:
            break
        
    sft = bundles_to_tgram(bundles, bundle_dict, read_mni_template())
    
    return sft


def _prealignment_centroids(base_dir, session_name, model_name, subjects, bundle_name, n_clusters):
    """
    calculate the centroid of each cluster for each subject.

    "prealignment" is meant to indicate that the clusters have not been aligned
    across models using a consensus subject. therefore the centroids are based
    on the "original" cluster labels determined by the subbundle model for each
    subject.

    Parameters
    ----------
    base_dir : str
    session_name : str
    model_name : str
    subjects : list
    bundle_name : str
    n_clusters : int

    Returns
    -------
    centroids : dict `subject_id` key
    └── containing an list of `StatefulTractograms` for each cluster
        with 100 nodes and mean xyz-coordinate in subject space
    """
    from os.path import join, exists
    import numpy as np
    from dipy.io.streamline import load_tractogram
    from dipy.tracking.streamline import set_number_of_points
    from dipy.io.stateful_tractogram import StatefulTractogram

    n_points = 100
    centroids = {}

    for subject in subjects:
        centroids[subject] = []
        subbundle_base_dir = join(base_dir, subject, session_name, str(n_clusters))

        for cluster_id in range(n_clusters):
            tractogram_name = f'{model_name}_cluster_{cluster_id}_clean.trk'
            tractogram_file = join(subbundle_base_dir, tractogram_name)

            if not exists(tractogram_file):
                logger.log(logging.INFO, f'{subject} {bundle_name} {cluster_id} not found')
                continue

            tractogram = load_tractogram(tractogram_file, 'same')

            centroid = np.mean(set_number_of_points(tractogram.streamlines, n_points), axis=0)

            logger.log(logging.DEBUG, f'generating centroid tractogram {subject} {bundle_name} {cluster_id}')
            centroid_sft = StatefulTractogram.from_sft([centroid], tractogram)
            centroids[subject].append(centroid_sft)

    return centroids

def _relabeled_centroids(base_dir, session_name, model_name, subjects, bundle_name, n_clusters, cluster_labels):
    """
    calculate the centroid of each cluster for each subject and order the clusters by the
    new cluster labels

    Parameters
    ----------
    base_dir : string
    session_name : string
    model_name : string
    subjects : array
    n_clusters : int
    cluster_labels : dict

    Returns
    -------
    centroids : dict
    """
    from os.path import exists, join
    import numpy as np
    from dipy.tracking.streamline import set_number_of_points
    from dipy.io.stateful_tractogram import StatefulTractogram
    from dipy.io.streamline import load_tractogram

    n_points = 100
    centroids = {}
    
    for subject in subjects:
        centroids[subject] = []

        working_dir = join(base_dir, subject, session_name, str(n_clusters))
        
        for cluster_label in range(n_clusters):
            cluster_id = cluster_labels[subject].tolist().index(cluster_label)
            clean_cluster_tractogram_filename = join(working_dir, f'{model_name}_cluster_{cluster_id}_clean.trk')

            if not exists(clean_cluster_tractogram_filename):
                logger.log(logging.INFO, f'{subject} {bundle_name} {cluster_id} not found')
                continue

            clean_cluster_tractogram = load_tractogram(clean_cluster_tractogram_filename, 'same', bbox_valid_check=False)

            centroid = np.mean(set_number_of_points(clean_cluster_tractogram.streamlines, n_points), axis=0)
            centroid_sft = StatefulTractogram.from_sft([centroid], clean_cluster_tractogram)
            centroids[subject].append(centroid_sft)

    return centroids

def _move_centroids_to_MNI(session_name, subjects, centroids):
    """
    move the centroids from subject space into MNI space

    Parameters
    ----------
    session_name : string
    subjects : array
    centroids : dict

    Returns
    -------
    mni_centroids : dict
    """
    mni_centroids = {}
    for subject in subjects:
        mni_centroids[subject] = []
        for centroid in centroids[subject]:
            logger.log(logging.DEBUG, 'moving centroid to MNI')
            mni_centroids[subject].append(move_tractogram_to_MNI_space(session_name, subject, centroid))
    
    return mni_centroids


def _match_subject_clusters_by_centroid_MDF(source, source_centroids, target, target_centroids):
    import numpy as np
    import itertools
    from dipy.tracking.streamline import bundles_distances_mdf
    from scipy.optimize import linear_sum_assignment

    pairwise_mdf = np.array([])
            
    # calculate distance between source and target centroids
    for (source_centroid, target_centroid) in itertools.product(source_centroids[source], target_centroids[target]):
        mdf = bundles_distances_mdf(source_centroid.streamlines, target_centroid.streamlines)
        pairwise_mdf = np.append(pairwise_mdf, mdf)

    pairwise_shape = (len(source_centroids[source]), len(target_centroids[target]))
    block = pairwise_mdf.reshape(pairwise_shape)

    # e.g.;
    # [[10.83614349 13.39884186 10.02115345]
    #  [20.7915287  25.46654701 18.38787651]
    #  [16.14920807 15.89604855 12.14775753]]

    # want column indicies
    _, target_labels = linear_sum_assignment(block)
    # e.g.; [0 2 1]

    logger.log(logging.DEBUG, f'centroid MDF {target} {source}\n{block}\n{target_labels}')

    return target_labels

def _match_clusters_by_centroid_MDF(base_dir, session_name, centroids, target, sources, n_clusters):
    """
    Given target's subbundle centroids assign target's labels to source's 
    subbundle centroids based on proximity, in MNI space, as calculated by MDF.

    Assignes each source centroid to an unique target label.
    """
    
    from os.path import join
    import numpy as np

    # For each subject pair compute the adjacency matrix
    for source in sources:
        target_labels = _match_subject_clusters_by_centroid_MDF(source, centroids, target, centroids)

        # align test-retest
        if session_name == "HCP_Retest":
            consensus_base_dir = join(base_dir, target, session_name, str(n_clusters))
            retest_labels = np.load(join(consensus_base_dir, f'consensus_mdf_labels.npy'))
            target_labels = _relabel_clusters(target_labels, retest_labels)

        source_subbundle_base_dir = join(base_dir, source, session_name, str(n_clusters))
        label_file = join(source_subbundle_base_dir, f'{target}_mdf_labels.npy')
        np.save(label_file, target_labels)


###############################################################################

def match_clusters(base_dir, session_name, subjects, cluster_info, target, n_clusters, algorithm):
    """
    Run matching algorithms for `target` on remaining `subjects` in dataset.

    New cluster labels are saved to disk. 
    
    target is the desired subject to match (depending on where are in pipeline this is candidate or consensus subject)
    """
    sources = subjects[:]
    sources.remove(target)

    if algorithm == Algorithm.MAXDICE:
        logger.log(logging.DEBUG, 'matching by maximum dice')
        _match_clusters_by_maximum_dice(
            base_dir,
            session_name,
            cluster_info[n_clusters][session_name]['tractograms'],
            cluster_info[n_clusters][session_name]['tractograms_filenames'],
            target,
            sources,
            n_clusters
        )
    elif algorithm == Algorithm.MUNKRES:
        logger.log(logging.DEBUG, 'matching by munkres')
        _match_clusters_by_munkres(
            base_dir,
            session_name,
            cluster_info[n_clusters][session_name]['tractograms'],
            cluster_info[n_clusters][session_name]['tractograms_filenames'],
            target,
            sources,
            n_clusters
        )
    elif algorithm == Algorithm.CENTROID:
        logger.log(logging.DEBUG, 'matching by centroid')
        _match_clusters_by_centroid_MDF(
            base_dir,
            session_name,
            cluster_info[n_clusters][session_name]['centroids'],
            target,
            sources,
            n_clusters
        )
    else:
        logger.log(logging.ERROR, 'unknown algorithm')


def _match_consensus_subject_test_retest_clusters(metadata, cluster_info, n_clusters):
    """
    align consensus subject clusters across test-retest sessions
    """
    from os.path import join
    import numpy as np

    base_dir = join(metadata['experiment_output_dir'], metadata['bundle_name'])
    consensus_base_dir = join(base_dir, cluster_info[n_clusters]['consensus_subject'], metadata['experiment_retest_session'], str(n_clusters))
    
    if metadata['algorithm'] == Algorithm.CENTROID:
        test_centroids = cluster_info[n_clusters][metadata['experiment_test_session']]['centroids']
        retest_centroids = cluster_info[n_clusters][metadata['experiment_retest_session']]['centroids']

        retest_labels = _match_subject_clusters_by_centroid_MDF(
            cluster_info[n_clusters]['consensus_subject'], test_centroids,
            cluster_info[n_clusters]['consensus_subject'], retest_centroids
        )

        np.save(join(consensus_base_dir, f'consensus_mdf_labels.npy'), retest_labels)
    else:
        test_tractograms = cluster_info[n_clusters][metadata['experiment_test_session']]['tractograms']
        test_tractograms_filenames = cluster_info[n_clusters][metadata['experiment_test_session']]['tractograms_filenames']

        retest_tractograms = cluster_info[n_clusters][metadata['experiment_retest_session']]['tractograms']
        retest_tractograms_filenames = cluster_info[n_clusters][metadata['experiment_retest_session']]['tractograms_filenames']


        if metadata['algorithm'] == Algorithm.MAXDICE:
            retest_labels = _match_subject_clusters_by_maximum_dice(
                cluster_info[n_clusters]['consensus_subject'], test_tractograms, test_tractograms_filenames,
                cluster_info[n_clusters]['consensus_subject'], retest_tractograms, retest_tractograms_filenames
            )

            np.save(join(consensus_base_dir, f'consensus_maxdice_labels.npy'), retest_labels)
        elif metadata['algorithm'] == Algorithm.MUNKRES:
            retest_labels = _match_subject_clusters_by_munkres(
                cluster_info[n_clusters]['consensus_subject'], test_tractograms, test_tractograms_filenames,
                cluster_info[n_clusters]['consensus_subject'], retest_tractograms, retest_tractograms_filenames
            )

            np.save(join(consensus_base_dir, f'consensus_munkres_labels.npy'), retest_labels)
        else:
            logger.log(logging.ERROR, 'unknown algorithm')


###############################################################################
###############################################################################
# Step 2:
# • calculate the variance in the FA profiles for new clusters
#   - expect this to be better than the original cluster labels which were ordered
#     by number of streamlines (counts) within each subject
###############################################################################
###############################################################################

class Scalars:
    DTI_FA = 'DTI_FA.nii.gz'
    DTI_MD = 'DTI_MD.nii.gz'

def _load_scalar_data(base_dir, session_name, subjects, scalar=Scalars.DTI_FA, csd=True):
    """
    Loads the FA scalar data for all subjects. By default assumes CSD.

    Returns a dictionary with `subject` as key.
    
    Looks for scalar file locally, if it does not exist the code will attempt
    to download from AWS the hcp_reliability single shell study.
    └── `{base_dir}/{subject}/{session_name}/{scalar}`
    """
    import s3fs
    from os.path import exists, join
    import nibabel as nib

    fs = s3fs.S3FileSystem()

    scalar_data = {}

    for subject in subjects:
        scalar_data[subject] = {}
        scalar_filename = join(base_dir, subject, session_name, scalar)
        if not exists(scalar_filename):
            logger.log(logging.DEBUG, f'downloading scalar {scalar_filename}')
            if csd:
                fs.get(
                    (
                        f'profile-hcp-west/hcp_reliability/single_shell/'
                        f'{session_name.lower()}_afq_CSD/'
                        f'sub-{subject}/ses-01/'
                        f'sub-{subject}_dwi_model-{scalar}'
                    ),
                    scalar_filename
                )
            else:
                fs.get(
                    (
                        f'profile-hcp-west/hcp_reliability/single_shell/'
                        f'{session_name.lower()}_afq/'
                        f'sub-{subject}/ses-01/'
                        f'sub-{subject}_dwi_model-{scalar}'
                    ),
                    scalar_filename
                )

        logger.log(logging.DEBUG, f'loading {scalar_filename}')
        scalar_data[subject] = nib.load(scalar_filename).get_fdata()

    return scalar_data


def _load_bundle_tractograms(base_dir, session_name, subjects, bundle_name, csd=True):
    """
    Loads the bundle tractogram for all `subjects`. Bundle is specified with
    `bundle_name`. By default assumes CSD.

    Returns a dictionary with `subject` as key.

    Looks for tractogram file locally, if does not exist the code will attempt
    to download from AWS the hcp_reliability single shell study.
    └── `{base_dir}/{subject}/{session_name}/{bundle_name}.trk'
    """
    import s3fs
    from os.path import exists, join
    from dipy.io.streamline import load_tractogram

    fs = s3fs.S3FileSystem()

    tractogram_basename = f'{bundle_name}.trk'

    tractograms = {}

    for subject in subjects:
        tractograms[subject] = {}
        
        tractogram_filename = join(base_dir, subject, session_name, tractogram_basename)

        if not exists(tractogram_filename):
            logger.log(logging.DEBUG, f'downloading tractogram {tractogram_filename}')
            if csd:
                fs.get(
                    (
                        f'profile-hcp-west/hcp_reliability/single_shell/'
                        f'{session_name.lower()}_afq_CSD/'
                        f'sub-{subject}/ses-01/'
                        f'clean_bundles/sub-{subject}_dwi_space-RASMM_model-CSD_desc-det-afq-{bundle_name}_tractography.trk'
                    ),
                    tractogram_filename
                )
            else:
                fs.get(
                    (
                        f'profile-hcp-west/hcp_reliability/single_shell/'
                        f'{session_name.lower()}_afq/'
                        f'sub-{subject}/ses-01/'
                        f'clean_bundles/sub-{subject}_dwi_space-RASMM_model-DTI_desc-det-afq-{bundle_name}_tractography.trk'
                    ),
                    tractogram_filename
                )
        
        logger.log(logging.DEBUG, f'loading {tractogram_filename}')
        tractogram = load_tractogram(tractogram_filename, 'same', bbox_valid_check=False)
        tractograms[subject] = tractogram

    return tractograms


def _get_relabeled_clusters(base_dir, session_name, subjects, n_clusters, target, algorithm):
    import numpy as np
    from os.path import join

    cluster_labels = {}

    for subject in subjects:
        if subject == target:
            cluster_labels[subject] = np.array(list(range(n_clusters)))
        else:
            cluster_file = join(base_dir, subject, session_name, str(n_clusters), f'{target}_{algorithm}_labels.npy')
            cluster_labels[subject] = np.load(cluster_file)
        
    return cluster_labels


def _get_cluster_subject_afq_profiles(base_dir, session_name, model_name, subjects, bundle_name, n_clusters, cluster_labels):
    """
    Calculate all afq profiles for each cluster. Where the cluster labels 
    originate from the cluster model `model_name`.

    NOTE: ONLY FA

    Returns two dictionaries
    
    one where the key is the cluster label and each value contains an array of
    arrays with each subjects weighted afq profile. used to calculate within
    cluster variance.

    one where the key is the subject id and each value contains an array of
    arrays with each clusters weighted afq profile. used to calculate across
    cluster variance.
    """
    from os.path import exists, join
    from dipy.io.streamline import load_tractogram
    from dipy.stats.analysis import afq_profile, gaussian_weights

    fa_scalar_data = _load_scalar_data(base_dir, session_name, subjects)

    # using tractograms - which have been transformed into MNI space is not going to work... why??
    # therefore need to load the cluster ids
    # bundle_tractograms = _load_bundle_tractograms(base_dir, session_name, subjects, bundle_name)
    
    cluster_profiles = {}

    for cluster_label in range(n_clusters):
        cluster_profiles[cluster_label] = []
    
    subject_profiles = {}

    for subject in subjects:
        subject_profiles[subject] = []

        # TODO does not work for MAXDICE!
        # For now just get working with MUNKRES and MDF

        working_dir = join(base_dir, subject, session_name, str(n_clusters))

        for cluster_label in range(n_clusters):
            cluster_id = cluster_labels[subject].tolist().index(cluster_label)
            clean_cluster_tractogram_filename = join(working_dir, f'{model_name}_cluster_{cluster_id}_clean.trk')

            if not exists(clean_cluster_tractogram_filename):
                logger.log(logging.INFO, f'{subject} {bundle_name} {cluster_id} not found')
                continue
            
            clean_cluster_tractogram = load_tractogram(clean_cluster_tractogram_filename, 'same', bbox_valid_check=False)

            cluster_profile = afq_profile(
                fa_scalar_data[subject],
                clean_cluster_tractogram.streamlines,
                clean_cluster_tractogram.affine,
                n_points=100,
                weights=gaussian_weights(
                    clean_cluster_tractogram.streamlines,
                    n_points=100
                )
            )

            cluster_profiles[cluster_label].append(cluster_profile)
            subject_profiles[subject].append(cluster_profile)

    return (cluster_profiles, subject_profiles)


def _get_bundle_afq_profiles(base_dir, session_name, subjects, bundle_name, scalar=Scalars.DTI_FA):
    """
    Calculate the weighted afq profile for each subjects bundle.

    Return array of arrays with each subjects weighted afq profile.
    """
    from os.path import join, exists
    from dipy.stats.analysis import afq_profile, gaussian_weights
    import numpy as np

    scalar_abr = scalar.split('.')[0]
    scalar_data = _load_scalar_data(base_dir, session_name, subjects, scalar)
    bundle_tractograms = _load_bundle_tractograms(base_dir, session_name, subjects, bundle_name)

    profiles = []

    for subject in subjects:
        bundle_profile_filename = join(base_dir, subject, session_name, f'{bundle_name}_{scalar_abr}_profile.npy')
        if not exists(bundle_profile_filename):
            logger.log(logging.DEBUG, f'generating bundle profile {subject} {bundle_name}')
            bundle_profile = afq_profile(
                scalar_data[subject],
                bundle_tractograms[subject].streamlines,
                bundle_tractograms[subject].affine,
                weights=gaussian_weights(bundle_tractograms[subject].streamlines)
            )

            logger.log(logging.DEBUG, f'saving {bundle_profile_filename}')
            np.save(bundle_profile_filename, bundle_profile)
        else:
            logger.log(logging.DEBUG, f'loading {bundle_profile_filename}')
            bundle_profile = np.load(bundle_profile_filename)
        
        profiles.append(bundle_profile)

    return profiles


def _get_profiles(base_dir, session_name, model_name, subjects, bundle_name, n_clusters, algorithm):
    """
    Calculating the bundle and original profiles are optional since not
    used to determine the consensus subject, however they are beneficial in
    assessing whether the cluterting or relabeling clustering improves performance

    returns the afq_profiles for bundle, original, and new
    """
    import numpy as np

    # bundle_profiles consists of (N subjects, 100 nodes)
    bundle_profiles = _get_bundle_afq_profiles(base_dir, session_name, subjects, bundle_name)
    
    # The original cluster lables is just the same ordered list for all subjects
    cluster_labels = {}
    for subject in subjects:
        cluster_labels[subject] = np.array(list(range(n_clusters)))

    (orig_cluster_profiles, orig_subject_profiles) = _get_cluster_subject_afq_profiles(
        base_dir, session_name, model_name, subjects, bundle_name, n_clusters, cluster_labels
    )

    new_cluster_profiles = {}
    new_subject_profiles = {}

    # WARNING: each cluster may have different number of profiles (N profiles, 100 nodes) if using Algorithm.MAXDICE
    for target in subjects:
        cluster_labels = _get_relabeled_clusters(base_dir, session_name, subjects, n_clusters, target, algorithm)

        (new_cluster_profiles[target], new_subject_profiles[target]) = _get_cluster_subject_afq_profiles(
            base_dir, session_name, model_name, subjects, bundle_name, n_clusters, cluster_labels
        )
        
    return (bundle_profiles, orig_cluster_profiles, orig_subject_profiles, new_cluster_profiles, new_subject_profiles)


def _calculate_ratios(base_dir, session_name, model_name, subjects, bundle_name, n_clusters, algorithm, calc_bundle=False, calc_original=False):
    """
    apply otsu's critria 
    - minimize variance within subbundles (intraclass)
    - maximize variance across subbundles (interclass)

    For each of the following FA afq profiles: 
    - bundle
    - original model clusters
    - new clusters based on candidate target subject. 

    Calculate 
    - intraclass statistic -- across clusters
    - interclass statistic -- across subjects
    - ratio intraclass/interclass
    """
    import numpy as np
    
    (
        bundle_profiles, 
        orig_cluster_profiles, orig_subject_profiles, 
        new_cluster_profiles, new_subject_profiles
    ) = _get_profiles(
        base_dir,
        session_name,
        model_name,
        subjects,
        bundle_name,
        n_clusters,
        algorithm
    )

    bundle_ratio = None

    if calc_bundle:
        ###################################
        # baseline comparison -- calculate ratio for bundle
        # to determine whether clustering beneficial
        # optional -- not used in determining consensus subject

        # calculate the mean of standard deviation of the fa profile for each subject's bundle.
        bundle_cluster_intraclass = np.mean(np.std(np.array(bundle_profiles), axis=0))
        bundle_subject_interclass = np.mean(np.std(np.array(bundle_profiles), axis=1))
        bundle_ratio = bundle_cluster_intraclass/bundle_subject_interclass
        
        logger.log(logging.DEBUG, 
            f'bundle {bundle_name}' +
            f' profile intra-class (bundle) {bundle_subject_interclass}' +
            f' profile inter-class (subjects) {bundle_subject_interclass}' +
            # calculate the ratio a pseudo F-score
            f' profile ratio {bundle_ratio}'
        )

    orig_ratio = None

    if calc_original:
        ###################################
        # baseline comparison -- calculate the original cluster ratios
        # to determine whether relabeling beneficial
        # optional -- not used in determining consensus subject

        # within cluster variation
        # calculate the mean of standard deviation of the fa profile for each subjects
        # cluster.
        orig_cluster_intraclass = []
        
        for cluster_name in range(n_clusters):
            orig_cluster_intraclass.append(np.mean(np.std(np.array(orig_cluster_profiles[cluster_name]), axis=0)))
        
        # take the mean of means across all clusters
        total_orig_cluster_intraclass = np.nanmean(orig_cluster_intraclass)

        # within subject variation
        total_orig_subject_interclass = np.mean(np.array([np.mean(np.std(np.array(orig_subject_profiles[subject]))) for subject in subjects]))

        orig_ratio = total_orig_cluster_intraclass/total_orig_subject_interclass

        logger.log(logging.DEBUG, 
            f'original clusters' +
            f' profile intra-class (cluster) {total_orig_cluster_intraclass}' +
            f' profile inter-class (subjects) {total_orig_subject_interclass}' +
            # calculate the ratio a pseudo F-score
            f' profile ratio {orig_ratio}'
        )

    ###################################
    # calculate the relabeled cluster ratios

    # track across subjects
    population_new_cluster_interclass = {}

    for cluster_name in range(n_clusters):
        population_new_cluster_interclass[cluster_name] = []
    
    population_total_new_cluster_interclass = []

    new_ratios = []

    for target in subjects:
        # within cluster variation
        # calculate the mean of standard deviation of the fa profile for each subjects
        # cluster. 
        new_cluster_intraclass = []
    
        for cluster_name in range(n_clusters):
            intraclass_statistic = np.mean(np.std(np.array(new_cluster_profiles[target][cluster_name]), axis=0))
            new_cluster_intraclass.append(intraclass_statistic)

            # store statistics by cluster
            population_new_cluster_interclass[cluster_name].append(intraclass_statistic)
        
        # take the mean of means across all clusters
        total_new_cluster_intraclass = np.nanmean(new_cluster_intraclass)
        
        # store subject statistic
        population_total_new_cluster_interclass.append(total_new_cluster_intraclass)

        # within subject variation
        total_new_subject_interclass = np.mean(np.array([np.mean(np.std(np.array(new_subject_profiles[target][subject]))) for subject in subjects]))

        new_ratio = total_new_cluster_intraclass/total_new_subject_interclass
        new_ratios.append(new_ratio)

        logger.log(logging.DEBUG, 
            f'relabeled clusters using {target} as consensus' +
            f' profile intra-class (cluster) {total_new_cluster_intraclass}' +
            f' profile inter-class (subjects) {total_new_subject_interclass}' +
            # calculate the ratio a pseudo F-score
            f' profile ratio {new_ratio}'
        )

    logger.log(logging.DEBUG, f'min/max total var {min(population_total_new_cluster_interclass)} {max(population_total_new_cluster_interclass)}')
    
    for cluster_name in range(n_clusters):
        logger.log(logging.DEBUG, f'min/max cluster {cluster_name} {min(population_new_cluster_interclass[cluster_name])} {max(population_new_cluster_interclass[cluster_name])}')

    return (bundle_ratio, orig_ratio, new_ratios)


###############################################################################
###############################################################################
# Step 3:
# • iterate changing target for each subject
# • choose the label with lowest variance in FA profiles call this the
#   "true" cluster label - i.e. subbundle
###############################################################################
###############################################################################


def _find_consensus_subject(base_dir, session_name, model_name, subjects, cluster_info, bundle_name, n_clusters, algorithm=Algorithm.MUNKRES):
    """
    Select and return `consensus_subject` for the `session_name` population of `subjects` based on the `algorithm`.

    A `consensus_subject` is the one with smallest difference between it's clusters and the population.

    WARNING: if the consensus_subject has different number of test-retest clusters, probably not
    ideal choice (Algorithm.MAXDICE)
    """

    for subject in subjects:
        logger.log(logging.DEBUG, f'matching clusters to subject {subject}')
        match_clusters(
            base_dir,
            session_name,
            subjects,
            cluster_info,
            subject,
            n_clusters,
            algorithm
        )
    
    _, _, new_ratios = _calculate_ratios(base_dir, session_name, model_name, subjects, bundle_name, n_clusters, algorithm)

    # choose the minimum profile ratio as consensus subject
    _, idx = min((val, idx) for (idx, val) in enumerate(new_ratios))

    consensus_subject = subjects[idx]
    logger.log(logging.INFO, f'{n_clusters} consensus subject {consensus_subject}')

    return consensus_subject


###############################################################################
###############################################################################
# Step 4:
# • Once identified 'consensus subject', repeat for each session 
#   Ensure using same target subject across sessions: no peaking!
# • align labels across sessions as before.
###############################################################################
###############################################################################

def clean(metadata):
    """
    clean function to remove files created by this script
    """
    import os
    
    # don't need to delete the MNI and densities
#     extensions = [f'{metadata["algorithm"]}_labels.npy', 'MNI.trk', 'density_map.nii.gz']
    
    extensions = [f'{metadata["algorithm"]}_labels.npy']
    
    for root, dirs, files in os.walk(os.path.join('subbundles', metadata['experiment_name'], metadata['bundle_name'])):
        for file in files:
            for extension in extensions:
                if file.endswith(extension):
                    os.remove(os.path.join(root, file))


# Now that have identified "consensus" subject (target)

def match_retest_clusters(metadata, cluster_info):
    """
    now that have identified the "consensus" subject -- match cluster retest
    """
    from os.path import join

    base_dir = join(metadata['experiment_output_dir'], metadata['bundle_name'])
    session_name = metadata['experiment_retest_session']

    for n_clusters in metadata['experiment_range_n_clusters']:
        _match_consensus_subject_test_retest_clusters(metadata, cluster_info, n_clusters)

        match_clusters(
            base_dir,
            session_name,
            metadata['experiment_subjects'],
            cluster_info,
            cluster_info[n_clusters]['consensus_subject'],
            n_clusters,
            metadata['algorithm']
        )

###############################################################################
###############################################################################
# Step 5: Choose K
###############################################################################
###############################################################################

def get_bundle_afq_profiles(metadata):
    """
    bundle_afq_profiles[scalar][subject][session_name][node_id]

    Returns
    -------
    bundle_afq_profiles : dict
    └── for each scalar in model return the scalar profile for the subject-session bundle
    """
    import itertools
    from os.path import join
    from dipy.stats.analysis import afq_profile, gaussian_weights

    base_dir = join(metadata['experiment_output_dir'], metadata['bundle_name'])

    bundle_afq_profiles = {}

    for scalar in metadata['model_scalars']:
        bundle_afq_profiles[scalar] = {}
        for subject in metadata['experiment_subjects']:
            bundle_afq_profiles[scalar][subject] = {}

    for scalar, session_name in itertools.product(metadata['model_scalars'],  metadata['experiment_sessions']):
        subject_session_afq_profiles = _get_bundle_afq_profiles(
            base_dir, 
            session_name, 
            metadata['experiment_subjects'],
            metadata['bundle_name'],
            scalar
        )

        # ensure dict keys is same as get_cluster_afq_profiles
        for subject in metadata['experiment_subjects']:
            bundle_afq_profiles[scalar][subject][session_name] = subject_session_afq_profiles[metadata['experiment_subjects'].index(subject)]

    return bundle_afq_profiles


def get_cluster_afq_profiles(metadata, n_clusters, target):
    """
    cluster_afq_profiles[scalar][subject][session_name][cluster_id][node_id]

    Returns
    -------
    cluster_afq_profiles : dict
    └── for each scalar in model return the scalar profile for the subject-session cluster
    """
    from os.path import exists, join
    from dipy.stats.analysis import afq_profile, gaussian_weights
    from dipy.io.streamline import load_tractogram

    base_dir = join(metadata['experiment_output_dir'], metadata['bundle_name'])
    model_name = metadata['model_name']
    bundle_name = metadata['bundle_name']

    cluster_afq_profiles = {}

    for scalar in metadata['model_scalars']:
        cluster_afq_profiles[scalar] = {}

        for subject in metadata['experiment_subjects']:
            cluster_afq_profiles[scalar][subject] = {}
            for session_name in metadata['experiment_sessions']:
                cluster_afq_profiles[scalar][subject][session_name] = {}
                scalar_data = _load_scalar_data(base_dir, session_name, metadata['experiment_subjects'], scalar)
                cluster_labels = _get_relabeled_clusters(base_dir, session_name, metadata['experiment_subjects'], n_clusters, target, metadata['algorithm'])
                working_dir = join(base_dir, subject, session_name, str(n_clusters))
                for cluster_label in range(n_clusters):
                    cluster_id = cluster_labels[subject].tolist().index(cluster_label)
                    clean_cluster_tractogram_filename = join(working_dir, f'{model_name}_cluster_{cluster_id}_clean.trk')

                    if not exists(clean_cluster_tractogram_filename):
                        logger.log(logging.INFO, f'{subject} {bundle_name} {cluster_id} not found')
                        continue

                    clean_cluster_tractogram = load_tractogram(clean_cluster_tractogram_filename, 'same', bbox_valid_check=False)
                
                    cluster_afq_profiles[scalar][subject][session_name][cluster_label] = afq_profile(
                        scalar_data[subject],
                        clean_cluster_tractogram.streamlines,
                        clean_cluster_tractogram.affine,
                        weights=gaussian_weights(clean_cluster_tractogram.streamlines)
                    )

    return cluster_afq_profiles


def get_bundle_profile_tensor(metadata, bundle_afq_profiles):
    """
    convert the `bundle_afq_profiles` dict into 1xNxMxS ndarray

    - $K$ is number of clusters (there is only 1 for the bundle)
    - $N=44$ is number of subjects,
    - $M=100$ is number of sampled streamline nodes, and 
    - $S=2$ is number of sessions
    """
    import itertools
    import numpy as np
    from os.path import join

    subjects = metadata['experiment_subjects']
    n_nodes = 100
    session_names = metadata['experiment_sessions']
    
    tensor = np.zeros((len(subjects), n_nodes, len(session_names)))

    for (subject, session, node_id) in itertools.product(subjects, session_names, range(n_nodes)):
        tensor[subjects.index(subject)][node_id][session_names.index(session)] = bundle_afq_profiles[subject][session][node_id]
        
    return np.asarray([tensor])


def get_cluster_profile_tensor(metadata, cluster_afq_profiles, n_clusters):
    """
    convert the `cluster_afq_profile` dict into KxNxMxS ndarray

    - $K$ is number of clusters,
    - $N=44$ is number of subjects,
    - $M=100$ is number of sampled streamline nodes, and 
    - $S=2$ is number of sessions
    """
    import itertools
    import numpy as np
    
    subjects = metadata['experiment_subjects']
    n_nodes = 100
    session_names = metadata['experiment_sessions']
    
    tensor = np.zeros((n_clusters, len(subjects), n_nodes, len(session_names)))

    for (subject, session, cluster_id, node_id) in itertools.product(subjects, session_names, range(n_clusters), range(n_nodes)):
        tensor[cluster_id][subjects.index(subject)][node_id][session_names.index(session)] = cluster_afq_profiles[subject][session][cluster_id][node_id]
    
    return tensor


def find_K(metadata, bundle_afq_profiles, cluster_afq_profiles):
    """
    Based on OTSU criteria:
    - Within bundle to be consistent - (intraclass)  
        _Want: the same subbundle to be same_
            - Across subjects within session - small variance
            - Within subjects across sessions - small variance
            - Across subjects across session - small variance

    - Across bundles to be different - (interclass)  
        _Want: different subbundles to be different_
            - Within subject within session - large variance
            - Across subjects within session - large variance
            - Within subjects across sessions - large variance
            - Across subjects across session - large variance

    use the minimum average root mean squared difference 
    between session scalar profiles 
    to determine the choice of K
    """
    import numpy as np

    for scalar in metadata['model_scalars']:
        avgRMSEs = []

        print(scalar, 'average RMSE root mean squared difference per subject\n')
        
        profile_tensor = get_bundle_profile_tensor(
            metadata,
            bundle_afq_profiles[scalar]
        )

        # take squared difference between sessions, then average across all nodes, then sqrt, and average across n_clusters
        avgRMSE = np.mean(np.sqrt(np.mean((profile_tensor[:,:,:,0]-profile_tensor[:,:,:,1])**2, axis=-1)), axis=0)
        avgRMSEs.append(avgRMSE)

        for n_clusters in metadata['experiment_range_n_clusters']:  
            profile_tensor = get_cluster_profile_tensor(
                metadata,
                cluster_afq_profiles[n_clusters][scalar],
                n_clusters
            )

            avgRMSE = np.mean(np.sqrt(np.mean((profile_tensor[:,:,:,0]-profile_tensor[:,:,:,1])**2, axis=-1)), axis=0)
            avgRMSEs.append(avgRMSE)

        data = np.array(avgRMSEs)
        
        # choose K with lowest average RMSE
        K = np.argmin(np.mean(data, axis=1))+1

        return K, data

###############################################################################
###############################################################################
# Step 6: Average Pseudo F-Score
###############################################################################
###############################################################################

def get_pseudo_f(metadata, cluster_afq_profiles, alpha=0.05):
    """
    display test-retest average pseudo F for each scalar profile and n_cluster
    """
    from IPython.display import display
    import pandas as pd
    import numpy as np
    from scipy.stats import f
        
    pseudo_f = {}

    dfs = []
    for n_clusters in metadata['experiment_range_n_clusters']:  
        pseudo_f[n_clusters] = {}
        for scalar in metadata['model_scalars']:
            profile_tensor = get_cluster_profile_tensor(
                metadata,
                cluster_afq_profiles[n_clusters][scalar],
                n_clusters
            )

            # between-cluster-sum-of-squares (cluster variance)
            bcss = np.mean(np.var(profile_tensor, axis=0, ddof=1), axis=0)

            # degree of freedom for numerator
            dfn = n_clusters - 1

            # within-cluster-sum-of-squares (subject variance)
            wcss = np.mean(np.var(profile_tensor, axis=1, ddof=n_clusters), axis=0)
            
            # degree of freedom for demoninator
            dfd = len(metadata['experiment_subjects']) - n_clusters
            
            crit = f.ppf(q=1-alpha, dfd=dfd, dfn=dfn)
            
            pseudoF = bcss/wcss

            pseudo_f[n_clusters][scalar] = pseudoF
            
            avg_pseudoF = np.mean(pseudoF, axis=0)

            dfs.append(pd.DataFrame({
                'crit value': crit,
                'avg pseudo F': np.mean(avg_pseudoF), 
                'significant?': np.mean(avg_pseudoF)>crit,
                'any session significant?': any(avg_pseudoF>crit),
                'any node significant?': np.any(pseudoF>crit)
            }, index=[n_clusters]))

    df = pd.DataFrame(pd.concat(dfs))
    display(df.style.set_caption("Pseudo F"))

    return pseudo_f

###############################################################################
###############################################################################
# Step 6: Reliability Plot
###############################################################################
###############################################################################

def get_bundle_dice_coefficients(metadata):
    """
    calculate the weighted dice coefficient for every subject's bundle.

    used to evaulate performance of cluster weighted dice coefficients.
    """

    from os.path import join
    from AFQ.utils.volume import dice_coeff

    base_dir = join(metadata['experiment_output_dir'], metadata['bundle_name'])

    bundle_dice_coef = {}

    test_sfts = _load_bundle_tractograms(
        base_dir,
        metadata['experiment_test_session'],
        metadata['experiment_subjects'],
        metadata['bundle_name']
    )
    
    retest_sfts = _load_bundle_tractograms(
        base_dir,
        metadata['experiment_retest_session'],
        metadata['experiment_subjects'],
        metadata['bundle_name']
    )

    for subject in metadata['experiment_subjects']:
        test_sft = test_sfts[subject]
        test_sft.to_vox()
        test_density_map = _get_density_map_img(test_sft)

        retest_sft = retest_sfts[subject]
        retest_sft.to_vox()
        retest_density_map = _get_density_map_img(retest_sft)

        bundle_dice_coef[subject] = dice_coeff(test_density_map, retest_density_map)

    return bundle_dice_coef

def get_cluster_dice_coefficients(metadata, cluster_info, n_clusters):
    from os.path import join
    import numpy as np
    from AFQ.utils.volume import dice_coeff

    cluster_dice_coef = {}

    base_dir = join(metadata['experiment_output_dir'], metadata['bundle_name'])

    test_cluster_labels = _get_relabeled_clusters(
        base_dir,
        metadata['experiment_test_session'],
        metadata['experiment_subjects'],
        n_clusters,
        cluster_info[n_clusters]['consensus_subject'],
        metadata['algorithm']
    )

    retest_cluster_labels = _get_relabeled_clusters(
        base_dir,
        metadata['experiment_retest_session'],
        metadata['experiment_subjects'],
        n_clusters,
        cluster_info[n_clusters]['consensus_subject'],
        metadata['algorithm']
    )

    for subject in metadata['experiment_subjects']:
        dice_coef_matrix = np.zeros((n_clusters, n_clusters))

        test_cluster_tractograms = cluster_info[n_clusters][metadata['experiment_test_session']]['tractograms'][subject]
        test_cluster_tractograms_filenames = cluster_info[n_clusters][metadata['experiment_test_session']]['tractograms_filenames'][subject]

        retest_cluster_tractograms = cluster_info[n_clusters][metadata['experiment_retest_session']]['tractograms'][subject]
        retest_cluster_tractograms_filenames = cluster_info[n_clusters][metadata['experiment_retest_session']]['tractograms_filenames'][subject]
        
        ii = 0
        jj = 0
        
        for test_cluster_label in range(n_clusters):
            test_cluster_id = test_cluster_labels[subject].tolist().index(test_cluster_label)
            test_cluster_density_map, _ = _get_density_map(test_cluster_tractograms[test_cluster_id], test_cluster_tractograms_filenames[test_cluster_id])

            for retest_cluster_label in range(n_clusters):
                retest_cluster_id = retest_cluster_labels[subject].tolist().index(retest_cluster_label)
                retest_cluster_density_map, _ = _get_density_map(retest_cluster_tractograms[retest_cluster_id], retest_cluster_tractograms_filenames[retest_cluster_id])
                
                dice_coef_matrix[ii][jj] = dice_coeff(test_cluster_density_map, retest_cluster_density_map)
                jj += 1
            ii += 1
            jj = 0

        cluster_dice_coef[subject] = np.diag(dice_coef_matrix)

    return cluster_dice_coef