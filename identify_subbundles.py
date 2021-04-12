"""
Now that have clusters for each subject, from one or more experiments/models,
(see subbundle_aws.ipynb):

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

###############################################################################
###############################################################################
# Step 0:
# • Move clusters into MNI space
###############################################################################
###############################################################################

def move_tractogram_to_MNI_space(data_dir, subject, tractogram):
    """
    For given subject and tractogram move that tractogram from subject space
    to MNI space using existing AFQ derivatives.

    NOTE: Looks for subjects DWI file
    
    `~/AFQ_data/{data_dir}/deriavatives/dmriprep/sub-{subject}/ses-01/dwi/sub-{subject}_dwi.nii.gz`

    if does not exist will attempt to download from HCP repository

    NOTE: Looks for MNI image

    `~/.cache/templateflow/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-01_T2w.nii.gz`

    NOTE: Looks for mapping DWI to MNI mapping file:

    `~/AFQ_data/{data_dir}/derivatives/afq/sub-{subject}/ses-01/sub-{subject}_dwi_mapping_from-DWI_to_MNI_xfm.nii.gz`

    if does not exist will attempt to download from AFQ HCP repository

    NOTE: Looks for prealign file:

    `~/AFQ_data/{data_dir}/derivatives/afq/sub-{subject}/ses-01/sub-{subject}_dwi_prealign_from-DWI_to-MNI_xfm.npy`

    if does not exist will attempt to download from AFQ HCP repository

    Parameters
    ----------
    data_dir : string
        HCP dataset identifier. either 'HCP_1200' or 'HCP_Retest'

    subject : string
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
    afq_derivatives_base_dir = join(afq_base_dir, data_dir, 'derivatives')

    ###############################################################################
    subject_path = join(f'sub-{subject}', 'ses-01')

    ###############################################################################
    # load subject dwi image

    dwi_derivatives_dir = join(afq_derivatives_base_dir, 'dmriprep')
    dwi_file = join(dwi_derivatives_dir, subject_path, 'dwi', f'sub-{subject}_dwi.nii.gz')

    # download HCP data for subject
    if not exists(dwi_file):
        logger.log(logging.DEBUG, f'downloading dwi {dwi_file}')
        afd.fetch_hcp([subject], study=data_dir)

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
                f'{data_dir.lower()}_afq_CSD/sub-{subject}/ses-01/'
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
                f'{data_dir.lower()}_afq_CSD/sub-{subject}/ses-01/'
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
    # NOTE: using prealign here not necessary
    # mapping = reg.read_mapping(mapping_file, dwi_img, MNI_T2_img, np.linalg.inv(np.load(prealign_file)))
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
    # NOTE: could combine into single transform operation, but not computationally intensive, 
    # and more explicit to separate i.e., current code is better for interpretability
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


def load_MNI_cluster_tractograms(base_dir, data_dir, model_name, subjects, bundle_name, n_clusters):
    """
    Ensure all cluster tractograms are in MNI for all subjects.

    Once subjects are in MNI space we can compare their clusters to determine
    similarity.
    
    NOTE: Caches (saves) the resulting MNI tractogram to disk. 
    Therefore, remove these files when using new model.

    `{base_dir}/{subject}/{data_dir}/{subject}_{bundle_name}_{cluster_id}_MNI.trk`

    NOTE: Looks for clean tractography file for each cluster. The `cluster_id`
    is based on the label assigned by the clustering model for that subject.
    As of current implementation these are ordered by number of streamlines.

    `{base_dir}/{subject}/{data_dir}/{model_name}_cluster_{cluster_id}_clean.trk`

    Parameters
    ----------
    base_dir : string
        base directory for the experiment.
    
    data_dir : string
        HCP dataset identifier. either 'HCP_1200' or 'HCP_Retest'

    model_name : string
        clustering model name. used to determine tractography files.
    
    subjects : array
        array of strings representing subject identifiers in the data set. can
        be a subset of subjects.
    
    bundle_name : string
        name of the bundle. correspond to pyAFQ bundle names

    n_clusters : integer
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

        subbundle_base_dir = join(base_dir, subject, data_dir, str(n_clusters))
        
        for cluster_id in range(n_clusters):
            MNI_tractogram_file = join(subbundle_base_dir, f'{subject}_{bundle_name}_{cluster_id}_MNI.trk')

            # load subject clusters into MNI space
            if not exists(MNI_tractogram_file):
                logger.log(logging.DEBUG, f'generating {MNI_tractogram_file}')
                tractogram_name = f'{model_name}_cluster_{cluster_id}_clean.trk'
                tractogram_file = join(subbundle_base_dir, tractogram_name)
                
                if not exists(tractogram_file):
                    logger.log(logging.DEBUG, f'{subject} {bundle_name} {cluster_id} not found')
                    continue

                tractogram = load_tractogram(tractogram_file, 'same')

                sft = move_tractogram_to_MNI_space(data_dir, subject, tractogram)

                ## Save tractogram to visually inspect in MI Brain
                # NOTE: remember to globally reinitize MI-Brain 
                # (otherwise streamlines may not be loaded correctly)
                # TODO: Ideally want to generate images using AFQ.viz.plotly

                # NOTE: some streamlines appears outside the brain boundary in the MNI template image
                logger.log(logging.DEBUG, f'saving {MNI_tractogram_file}')
                save_tractogram(sft, MNI_tractogram_file, bbox_valid_check=False)
            else:
                logger.log(logging.DEBUG, f'loading {MNI_tractogram_file}')
                sft = load_tractogram(MNI_tractogram_file, 'same')

            tractograms[subject].append(sft)
            tractograms_filenames[subject].append(MNI_tractogram_file)
    
    return tractograms, tractograms_filenames


###############################################################################
###############################################################################
# Step 1:
# • relabel clusters assuming a single fixed target subject
###############################################################################
###############################################################################

def get_dice_coefficients(source_sft, source_filename, target_sft, target_filename):
    """
    Given two tractograms determine the similary by calculating the weighted
    dice coefficient. Assumes source and target are colocated in same space.

    Parameters
    ----------
    source_sft : StatefulTractogram

    target_sft : StatefulTractogram

    tractograms_filenames : dict

    Returns
    -------
    dice_coeff : int
        weighted dice coefficient
    """
    from AFQ.utils.volume import dice_coeff
    from subbundle_model_analysis_utils import get_density_map

    source_sft.to_vox()
    source_map, _ = get_density_map(source_sft, source_filename)

    target_sft.to_vox()
    target_map, _ = get_density_map(target_sft, target_filename)

    logger.log(logging.DEBUG, 'calculating dice')
    return dice_coeff(source_map, target_map)


def relabel_clusters(labeled_clusters, new_labels):
    """
    Take original streamline cluster labels and return a new array with the 
    reassigned labels. Labels are initially assigned by the clustering model
    and saved to disk with suffix 'idx'. Since each cluster model is run per
    subject, want to relabel the subjects clusters to match across subjects.
    
    Parameters
    ----------
    labeled_clusters : array
        array of cluster labels. where index corresponds to the streamline 
        index in the bundle tractogram, and values correspond to the orignal
        cluter labels assigned by the subjects clustering model.
    
    new_labels : array
        array of cluster labels. where index corresponds to the original label
        and the value is the new desired label.

    Returns
    -------
    relabeled_clusters : array
        array of relabeled clusters. where index corresponds to the streamline
        index in the bundle tractogram, and values correspond to the new cluter
        labels.
    """
    import numpy as np
    
    logger.log(logging.DEBUG, 'relabeling clusters')

    label_map = dict(zip(np.arange(len(new_labels)), new_labels))

    relabeled_clusters = np.copy(labeled_clusters)

    for original_label, new_label in label_map.items():
        relabeled_clusters[labeled_clusters == original_label] = new_label

    return relabeled_clusters


def match_clusters_by_maximum_dice(base_dir, data_dir, model_name, tractograms, tractograms_filenames, target, sources, n_clusters):
    """
    Once every subjects clusters are located in MNI space, pairwise compare
    the weighted dice coefficents of each cluster. 
    
    This version of the algorithm maps each source cluster to the best target
    cluster that is the one with maximum dice coefficient. 
    
    NOTE: This means that the number of clusters is not preserved, and may 
    collapse multiple clusters identified by the model into a single cluster.
    
    For example three source clusters might all map to a single target cluster.

    NOTE: caches MxN adjacency block, where M and N are <= K and are the number
    of cluster in the target and source subject models. the elements of the matrix
    are the pairwise weighted dice coefficients.

    `{base_dir}/{target}/{data_dir}/{target}_{source}_adjacency_block.npy`

    NOTE: caches the cluster labels. an array of labels representing the maximal
    dice for source's cluster. thus it is possible to map all sources to single
    target cluster.

    `{base_dir}/{target}/{data_dir}/{target}_{source}_cluster_labels.npy`

    NOTE: creates a streamline index file with the new cluster labels for each source

    `{base_dir}/{source}/{data_dir}/{target}_idx.npy'

    NOTE: depends on the existence of an original streamline index file for the source

    `{base_dir}/{source}/{data_dir}/{model_name}_idx.npy`

    For each source subject saves the pairwise dice block, the new cluster 
    labels relative to target subject, and saves the corresponding idx file
    with relabeled streamlines 
    """
    from os.path import join, exists
    import itertools
    import numpy as np

    target_subbundle_base_dir = join(base_dir, target, data_dir, str(n_clusters))

    # use remaining subjects for pairwise comparison
    for source in sources:
        # adjacency_block_file = join(target_subbundle_base_dir, f'{target}_{source}_adjacency_block.npy')
        cluster_labels_file = join(target_subbundle_base_dir, f'{target}_{source}_maxdice_cluster_labels.npy')

        if not exists(cluster_labels_file):
            logger.log(logging.DEBUG, f'generating dice labels {cluster_labels_file}')
            pairwise_dice = np.array([])
            
            # compare source and target clusters using weighted dice
            # for ((source_tractogram, source_filename), (target_tractogram, target_filename)) in itertools.product((tractograms[source], tractograms_filenames[source]), (tractograms[target], tractograms_filenames[target])):
            for ((source_tractogram, target_tractogram), (source_filename, target_filename)) in zip(
                list(itertools.product(tractograms[source], tractograms[target])), 
                list(itertools.product(tractograms_filenames[source], tractograms_filenames[target]))):
                pairwise_dice = np.append(pairwise_dice, get_dice_coefficients(source_tractogram, source_filename, target_tractogram, target_filename))

            pairwise_shape = (len(tractograms[source]), len(tractograms[target]))
            block = pairwise_dice.reshape(pairwise_shape)

            # NOTE: could put the block into a dict with key source:target
            # then when looking for subject pairs
            # if key doesn't exist, then 
            # reverse subject order in pair and if exists then 
            # need to transpose block

            # save block (NOTE: for debugging)
            # np.save(adjacency_block_file, block)

            logger.log(logging.DEBUG, f'adjacency block {source} {target} {block}')
            
            # take the source and move label to correspond to target cluster with highest overlap
            # relabel clusters: i -> ids[i]
            # [1 0 0] => 0 -> 1, 1 -> 0, 2 -> 0 # subject 1 to subject 0
            # [0 0 0] => 0 -> 0, 1 -> 0, 2 -> 0 # subject 2 to subject 0
            # [1 2 0] => 1 -> 0, 1 -> 2, 2 -> 0 # subject 2 to subject 1
            target_labels = np.argmax(block, axis=1)
            # logger.log(logging.DEBUG, target_labels)

            # save relabeling
            logger.log(logging.DEBUG, f'saving {cluster_labels_file}')
            np.save(cluster_labels_file, target_labels)

            # logger.log(logging.DEBUG, block.T)
            # # [2 0] => 0 -> 2, 1 -> 0 # subject 0 to subject 1
            # # [0 2] => 0 -> 0, 1 -> 2 # subject 0 to subject 2
            # # [0 1 1] => 0 -> 0, 1 -> 1, 2 -> 1 # subject 1 to subject 2
            # logger.log(logging.DEBUG, np.argmax(block.T, axis=1))
            # logger.log(logging.DEBUG, np.argmax(block, axis=0)) # alternatively
        else:
            logger.log(logging.DEBUG, f'loading {cluster_labels_file}')
            target_labels = np.load(cluster_labels_file)

        # load original source cluster labels and create new source cluter label file reindexed with target labels
        # this can be used with the bundle tractogram to load clusters
        source_subbundle_base_dir = join(base_dir, source, data_dir, str(n_clusters))
        original_cluster_labels = np.load(join(source_subbundle_base_dir, f'{model_name}_idx.npy'))
        new_cluster_labels = relabel_clusters(original_cluster_labels, target_labels)
        index_file = join(source_subbundle_base_dir, f'{target}_maxdice_idx.npy')
        logger.log(logging.DEBUG, f'saving {index_file}')
        np.save(index_file, new_cluster_labels)

###############################################################################
# Find permutation of matrix that maximizes its trace using the Munkres algorithm.
# Source: https://gist.github.com/lebedov/9fa8b5a02a0e764cd40c
# Reference: https://stat.ethz.ch/pipermail/r-help/2010-April/236664.html
###############################################################################
def maximize_trace(a):
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

def match_clusters_by_best_remaining_dice(base_dir, data_dir, model_name, tractograms, tractograms_filenames, target, sources, n_clusters):
    """
    Once every subjects clusters are located in MNI space, pairwise compare
    the wieghted dice coefficents of each cluster.
    
    NOTE: alternative to match_clusters_by_maximum_dice labeling algorithm 
    Ensures clusters remain unique. The number of clusters of resulting
    clusters for each pair will be the minimum of number of target clusters
    and number of source clusters.

    NOTE: caches MxN adjacency block, where M and N are <= K and are the number
    of cluster in the target and source subject models. the elements of the matrix
    are the pairwise weighted dice coefficients.

    `{base_dir}/{target}/{data_dir}/{target}_{source}_adjacency_block.npy`

    NOTE: caches the cluster labels. an array of labels representing the maximal trace using
    pairwise weighed dice coefficent. each cluster in subject should map to unque cluster in
    target. with execption of when M>N.

    `{base_dir}/{target}/{data_dir}/{target}_{source}_munkres_cluster_labels.npy`

    NOTE: creates a streamline index file with the new cluster labels for each source

    `{base_dir}/{source}/{data_dir}/{target}_munkres_idx.npy'

    NOTE: depends on the existence of an original streamline index file for the source

    `{base_dir}/{source}/{data_dir}/{model_name}_idx.npy`
    """
    from os.path import join, exists
    import itertools
    import numpy as np

    target_subbundle_base_dir = join(base_dir, target, data_dir, str(n_clusters))

    # use remaining subjects for pairwise comparison
    for source in sources:
        # adjacency_block_file = join(target_subbundle_base_dir, f'{target}_{source}_adjacency_block.npy')
        munkres_cluster_labels_file = join(target_subbundle_base_dir, f'{target}_{source}_munkres_cluster_labels.npy')

        if not exists(munkres_cluster_labels_file):
            logger.log(logging.DEBUG, f'generating munkres labels {munkres_cluster_labels_file}')
            pairwise_dice = np.array([])

            # compare source and target clusters using weighted dice
            # for ((source_tractogram, source_filename), (target_tractogram, target_filename)) in itertools.product((tractograms[source], tractograms_filenames[source]), (tractograms[target], tractograms_filenames[target])):
            for ((source_tractogram, target_tractogram), (source_filename, target_filename)) in zip(
                list(itertools.product(tractograms[source], tractograms[target])), 
                list(itertools.product(tractograms_filenames[source], tractograms_filenames[target]))):
                pairwise_dice = np.append(pairwise_dice, get_dice_coefficients(source_tractogram, source_filename, target_tractogram, target_filename))

            pairwise_shape = (len(tractograms[source]), len(tractograms[target]))
            block = pairwise_dice.reshape(pairwise_shape)

            # NOTE: could put the block into a dict with key source:target
            # then when looking for subject pairs
            # if key doesn't exist, then 
            # reverse subject order in pair and if exists then 
            # need to transpose block

            # save block (NOTE: for debugging)
            # logger.log(logging.DEBUG, f'saving adjacency block {adjacency_block_file}')
            # np.save(adjacency_block_file, block)
            logger.log(logging.DEBUG, f'adjacency block {source} {target} {block}')

            target_labels = maximize_trace(block)

            logger.log(logging.DEBUG, f'saving {munkres_cluster_labels_file}')
            np.save(munkres_cluster_labels_file, target_labels)

        else:
            logger.log(logging.DEBUG, f'loading {munkres_cluster_labels_file}')
            target_labels = np.load(munkres_cluster_labels_file, allow_pickle=True)

        # load original source cluster labels and create new source cluter label file reindexed with target labels
        # this can be used with the bundle tractogram to load clusters
        source_subbundle_base_dir = join(base_dir, source, data_dir, str(n_clusters))
        original_cluster_labels = np.load(join(source_subbundle_base_dir, f'{model_name}_idx.npy'))
        new_cluster_labels = relabel_clusters(original_cluster_labels, target_labels)
        munkres_index_file = join(source_subbundle_base_dir, f'{target}_munkres_idx.npy')
        logger.log(logging.DEBUG, f'saving {munkres_index_file}')
        np.save(munkres_index_file, new_cluster_labels)


###############################################################################
# Since relabeling using weighted DICE coefficients is underperforming, 
# try using centriod MDF
###############################################################################

def prealignment_centroids(base_dir, data_dir, model_name, subjects, bundle_name, n_clusters):
    """
    calculate the centriod of each cluster for each subject

    Parameters
    ----------
    base_dir : string
    data_dir : string
    model_name : string
    subjects : array
    bundle_name : string
    n_clusters : integer

    Returns
    -------
    centroids : dict
        dictionary with subject as key containing an array for each cluster
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
        subbundle_base_dir = join(base_dir, subject, data_dir, str(n_clusters))

        for cluster_id in range(n_clusters):
            tractogram_name = f'{model_name}_cluster_{cluster_id}_clean.trk'
            tractogram_file = join(subbundle_base_dir, tractogram_name)

            if not exists(tractogram_file):
                logger.log(logging.INFO, f'{subject} {bundle_name} {cluster_id} not found')
                continue

            tractogram = load_tractogram(tractogram_file, 'same')

            centroid = np.mean(set_number_of_points(tractogram.streamlines, n_points), axis=0)

            logger.log(logging.DEBUG, f'generating centroid tractogram {subject} {bundle_name} {cluster_id}')
            centriod_sft = StatefulTractogram.from_sft([centroid], tractogram)
            centroids[subject].append(centriod_sft)

    return centroids

def relabled_centriods(base_dir, data_dir, subjects, bundle_name, cluster_idxs, cluster_names):
    """
    calculate the centroid of each cluster for each subject and order the clusters by the
    new cluster names

    Parameters
    ----------
    base_dir : string
    data_dir : string
    subjects : array
    bundle_name : string 
    cluster_idxs : dict
    cluster_names : dict

    Returns
    -------
    centroids : dict
    """
    import numpy as np
    from dipy.tracking.streamline import set_number_of_points
    from dipy.io.stateful_tractogram import StatefulTractogram

    n_points = 100
    centroids = {}
    
    bundle_tractograms = load_bundle_tractograms(base_dir, data_dir, subjects, bundle_name)

    for subject in subjects:
        centroids[subject] = []
        for cluster_name, cluster_idx in zip(cluster_names[subject], cluster_idxs[subject]):
            centroid = np.mean(set_number_of_points(bundle_tractograms[subject].streamlines[cluster_idx], n_points), axis=0)
            
            logger.log(logging.DEBUG, f'generating relabeled centroid tractogram {subject} {bundle_name} {cluster_name}')
            centriod_sft = StatefulTractogram.from_sft([centroid], bundle_tractograms[subject])
            centroids[subject].append(centriod_sft)

    return centroids

def move_centriods_to_MNI(data_dir, subjects, centroids):
    """
    move the centroids from subject space into MNI space

    Parameters
    ----------
    data_dir : string
    subjects : array
    centroids : dict

    Returns
    -------
    mni_centriods : dict
    """
    mni_centriods = {}
    for subject in subjects:
        mni_centriods[subject] = []
        for centroid in centroids[subject]:
            logger.log(logging.DEBUG, 'moving centroid to MNI')
            mni_centriods[subject].append(move_tractogram_to_MNI_space(data_dir, subject, centroid))
    
    return mni_centriods

def match_clusters_by_centroid_MDF(base_dir, data_dir, model_name, centriods, target, sources, n_clusters):
    """
    Given target's subbundle centroids assign target's labels to source's 
    subbundle centroids based on proximity, in MNI space, as calculated by MDF.

    Assignes each source centroid to an unique target label.
    """
    import numpy as np
    import itertools
    from dipy.tracking.streamline import bundles_distances_mdf
    from os.path import join

    # For each subject pair compute the adjacency matrix
    for source in sources:
        pairwise_mdf = np.array([])
            
        # calculate distance between source and target centroids
        for (source_centriod, target_centroid) in itertools.product(centriods[source], centriods[target]):
            mdf = bundles_distances_mdf(source_centriod.streamlines, target_centroid.streamlines)
            pairwise_mdf = np.append(pairwise_mdf, mdf)

        pairwise_shape = (len(centriods[source]), len(centriods[target]))
        block = pairwise_mdf.reshape(pairwise_shape)
        # logger.log(logging.DEBUG, f'{target} {source} {block}')

        # TODO want to minimize the trace 
        target_labels = np.argmax(block, axis=1)

        source_subbundle_base_dir = join(base_dir, source, data_dir, str(n_clusters))
        original_cluster_labels = np.load(join(source_subbundle_base_dir, f'{model_name}_idx.npy'))
        new_cluster_labels = relabel_clusters(original_cluster_labels, target_labels)
        logger.log(logging.DEBUG, f'{target} {source} {new_cluster_labels}')
        centriod_index_file = join(source_subbundle_base_dir, f'{target}_mdf_idx.npy')
        logger.log(logging.DEBUG, f'saving {centriod_index_file}')
        np.save(centriod_index_file, new_cluster_labels)

###############################################################################

class Algorithm:
    """
    enum representing supported aglorithms to find matching clustering labels
    across models
    """
    MAXDICE = 'maxdice'
    MUNKRES = 'munkres'
    CENTROID = 'centroid'

def match_clusters(base_dir, data_dir, model_name, subjects, tractograms, tractograms_filenames, target, n_clusters, algorithm):
    """
    Run matching algorithms for `target` on remaining `subjects` in dataset.

    New cluster labels are saved to disk. 
    
    See `match_clusters_by_maximum_dice`, `match_clusters_by_best_remaining_dice`,
    and `match_clusters_by_centroid_MDF` for details.
    """
    sources = subjects[:]
    sources.remove(target)

    if algorithm == Algorithm.MAXDICE:
        logger.log(logging.DEBUG, 'matching by maximum dice')
        match_clusters_by_maximum_dice(base_dir, data_dir, model_name, tractograms, tractograms_filenames, target, sources, n_clusters)
    elif algorithm == Algorithm.MUNKRES:
        logger.log(logging.DEBUG, 'matching by munkres')
        match_clusters_by_best_remaining_dice(base_dir, data_dir, model_name, tractograms, tractograms_filenames, target, sources, n_clusters)
    elif algorithm == Algorithm.CENTROID:
        logger.log(logging.DEBUG, 'matching by centriod')
        match_clusters_by_centroid_MDF(base_dir, data_dir, model_name, tractograms, target, sources, n_clusters)
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

def load_fa_scalar_data(base_dir, data_dir, subjects, csd=True):
    """
    Loads the FA scalar data for all subjects. By default assumes CSD.

    Returns a dictionary with `subject` as key.
    
    NOTE: Looks for scalar file locally, if it does not exist the code will attempt
    to download from AWS the hcp_reliability single shell study.

    `{base_dir}/{subject}/{data_dir}/FA.nii.gz`
    """
    import s3fs
    from os.path import exists, join
    import nibabel as nib

    fs = s3fs.S3FileSystem()

    scalar_basename = 'FA.nii.gz'

    scalar_data = {}

    for subject in subjects:
        scalar_data[subject] = {}
        scalar_filename = join(base_dir, subject, data_dir, scalar_basename)
        if not exists(scalar_filename):
            logger.log(logging.DEBUG, f'downloading scalar {scalar_filename}')
            if csd:
                fs.get(
                    (
                        f'profile-hcp-west/hcp_reliability/single_shell/'
                        f'{data_dir.lower()}_afq_CSD/'
                        f'sub-{subject}/ses-01/'
                        f'sub-{subject}_dwi_model-DTI_FA.nii.gz'
                    ),
                    scalar_filename
                )
            else:
                fs.get(
                    (
                        f'profile-hcp-west/hcp_reliability/single_shell/'
                        f'{data_dir.lower()}_afq/'
                        f'sub-{subject}/ses-01/'
                        f'sub-{subject}_dwi_model-DTI_FA.nii.gz'
                    ),
                    scalar_filename
                )

        logger.log(logging.DEBUG, f'loading {scalar_filename}')
        scalar_data[subject] = nib.load(scalar_filename).get_fdata()

    return scalar_data


def load_bundle_tractograms(base_dir, data_dir, subjects, bundle_name, csd=True):
    """
    Loads the bundle tractogram for all `subjects`. Bundle is specified with
    `bundle_name`. By default assumes CSD.

    Returns a dictionary with `subject` as key.

    NOTE: Looks for tractogram file locally, if does not exist the code will attempt
    to download from AWS the hcp_reliability single shell study.

    `{base_dir}/{subject}/{data_dir}/{bundle_name}.trk'
    """
    import s3fs
    from os.path import exists, join
    from dipy.io.streamline import load_tractogram

    fs = s3fs.S3FileSystem()

    tractogram_basename = f'{bundle_name}.trk'

    tractograms = {}

    for subject in subjects:
        tractograms[subject] = {}
        
        tractogram_filename = join(base_dir, subject, data_dir, tractogram_basename)

        if not exists(tractogram_filename):
            logger.log(logging.DEBUG, f'downloading tractogram {tractogram_filename}')
            if csd:
                fs.get(
                    (
                        f'profile-hcp-west/hcp_reliability/single_shell/'
                        f'{data_dir.lower()}_afq_CSD/'
                        f'sub-{subject}/ses-01/'
                        f'clean_bundles/sub-{subject}_dwi_space-RASMM_model-CSD_desc-det-afq-{bundle_name}_tractography.trk'
                    ),
                    tractogram_filename
                )
            else:
                fs.get(
                    (
                        f'profile-hcp-west/hcp_reliability/single_shell/'
                        f'{data_dir.lower()}_afq/'
                        f'sub-{subject}/ses-01/'
                        f'clean_bundles/sub-{subject}_dwi_space-RASMM_model-DTI_desc-det-afq-{bundle_name}_tractography.trk'
                    ),
                    tractogram_filename
                )
        
        logger.log(logging.DEBUG, f'loading {tractogram_filename}')
        tractogram = load_tractogram(tractogram_filename, 'same', bbox_valid_check=False)
        tractograms[subject] = tractogram

    return tractograms


def load_labeled_clusters(base_dir, data_dir, model_name, subjects, n_clusters):
    """
    Loads the clusters from clustering model for each `subject`. The file 
    contains an ndarray with a cluster label, 0...K-1, for each streamline.
    The ndarray index corresponds to the streamline index in the bundle
    tractogram.

    Returns a tuple containing two dictionaries with `subject` as keys.
    The first dictionary contains an array of arrays of streamline indexes
    for each cluster label.
    The second dictionary contains an array of the cluster labels

    NOTE: looks for the original cluster labels locally in

    `./{base_dir}/{subject}/{data_dir}/{model_name}_idx.npy`

    see `vizualizations.load_clusters`
    """
    import numpy as np
    from os.path import join

    cluster_idxs = {}
    cluster_names = {}

    for subject in subjects:
        cluster_file = join(base_dir, subject, data_dir, str(n_clusters), f'{model_name}_idx.npy')
        logger.log(logging.DEBUG, f'loading clusters {cluster_file}')
        cluster_labels = np.load(cluster_file)
        cluster_names[subject] = np.unique(cluster_labels)
        cluster_idxs[subject] = np.array([np.where(cluster_labels == i)[0] for i in np.unique(cluster_labels)])
    
    return (cluster_idxs, cluster_names)


def load_relabeled_clusters(base_dir, data_dir, model_name, subjects, n_clusters, target, algorithm='munkres'):
    """
    Load the subjects clusters using labels from target subject.
    If the subject is the target then loads the original labels.
    By default assumes using Munkres algorithm to determine
    labeling strategy.

    NOTE: Looks for following files:

    `./{base_dir}/{subject}/{data_dir}/{model_name}_idx.npy`
    `./{base_dir}/{subject}/{data_dir}/{target}_munkres_idx.npy`
    `./{base_dir}/{subject}/{data_dir}/{target}_mdf_idx.npy`
    `./{base_dir}/{subject}/{data_dir}/{target}_idx.npy`

    see `load_labeled_clusters`
    TODO: duplicates much of logic; combine with load_labeled_clusters
    """
    import numpy as np
    from os.path import join

    cluster_idxs = {}
    cluster_names = {}

    for subject in subjects:
        if subject == target:
            cluster_filename = f'{model_name}_idx.npy'
        elif algorithm is None:
            cluster_filename = f'{target}_idx.npy'
        else:
            cluster_filename = f'{target}_{algorithm}_idx.npy'

        cluster_file = join(base_dir, subject, data_dir, str(n_clusters), cluster_filename)
        logger.log(logging.DEBUG, f'loading relabeled clusters {cluster_file}')
        cluster_labels = np.load(cluster_file)
        cluster_names[subject] = np.unique(cluster_labels)
        cluster_idxs[subject] = np.array([np.where(cluster_labels == i)[0] for i in np.unique(cluster_labels)])
    
    return (cluster_idxs, cluster_names)


def get_cluster_subject_afq_profiles(base_dir, data_dir, subjects, bundle_name, cluster_idxs, cluster_names, n_clusters):
    """
    Calculate all afq profiles for each cluster. Where the cluster labels 
    originate from the cluster model `model_name`.

    Returns two dictionaries
    
    one where the key is the cluster label and each value contains an array of
    arrays with each subjects weighted afq profile. used to calculate within
    cluster variance.

    one where the key is the subject id and each value contains an array of
    arrays with each clusters weighted afq profile. used to calculate across
    cluster variance.
    """
    # subbundle profiles from original custer labels
    from os.path import join, exists
    from dipy.stats.analysis import afq_profile, gaussian_weights
    import numpy as np

    fa_scalar_data = load_fa_scalar_data(base_dir, data_dir, subjects)

    # using tractograms - which have been transformed into MNI space is not going to work...
    # therefore need to load the cluster ids
    bundle_tractograms = load_bundle_tractograms(base_dir, data_dir, subjects, bundle_name)
    
    cluster_profiles = {}

    for cluster_name in range(n_clusters):
        cluster_profiles[cluster_name] = []
    
    subject_profiles = {}

    for subject in subjects:
        subject_profiles[subject] = []
        for cluster_name, cluster_idx in zip(cluster_names[subject], cluster_idxs[subject]):
            cluster_profile_filename = join(base_dir, subject, data_dir, str(n_clusters), f'{bundle_name}_cluster_{cluster_name}_profile.npy')
            if not exists(cluster_profile_filename):
                logger.log(logging.DEBUG, f'generating cluster profile {subject} {bundle_name} {cluster_name}')
                cluster_profile = afq_profile(
                    fa_scalar_data[subject],
                    bundle_tractograms[subject].streamlines[cluster_idx],
                    bundle_tractograms[subject].affine,
                    weights=gaussian_weights(bundle_tractograms[subject].streamlines[cluster_idx])
                )

                logger.log(logging.DEBUG, f'saving {cluster_profile_filename}')
                np.save(cluster_profile_filename, cluster_profile)
            else:
                logger.log(logging.DEBUG, f'loading {cluster_profile_filename}')
                cluster_profile = np.load(cluster_profile_filename)

            cluster_profiles[cluster_name].append(cluster_profile)
            subject_profiles[subject].append(cluster_profile)

    return (cluster_profiles, subject_profiles)


def get_bundle_afq_profiles(base_dir, data_dir, subjects, bundle_name):
    """
    Calculate the weighted afq profile for each subjects bundle.

    Return array of arrays with each subjects weighted afq profile.
    """
    from os.path import join, exists
    from dipy.stats.analysis import afq_profile, gaussian_weights
    import numpy as np

    fa_scalar_data = load_fa_scalar_data(base_dir, data_dir, subjects)
    bundle_tractograms = load_bundle_tractograms(base_dir, data_dir, subjects, bundle_name)

    profiles = []

    for subject in subjects:
        bundle_profile_filename = join(base_dir, subject, data_dir, f'{bundle_name}_profile.npy')
        if not exists(bundle_profile_filename):
            logger.log(logging.DEBUG, f'generating bundle profile {subject} {bundle_name}')
            bundle_profile = afq_profile(
                fa_scalar_data[subject],
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


def get_profiles(base_dir, data_dir, model_name, subjects, bundle_name, n_clusters):
    """
    NOTE:
    1) When using graspologic, depending on number of clusters 
        found, some subjects will have fewer then n_clusters, and 
    2) The choice of relabeling algorithm the number of subjects clusters may
        merge (as in the non-Munkres case).

    NOTE: calculating the bundle and original profiles are optional since not
    used to determine the consensus subject, however they are beneficial in 
    assessing whether the cluterting or relabeling clustering improves performance

    returns the afq_profiles for bundle, original, and new 
    """

    # bundle_profiles consists of (N subjects, 100 nodes)
    bundle_profiles = get_bundle_afq_profiles(base_dir, data_dir, subjects, bundle_name)
    
    # dictionary of K clusters with N profiles, where N <= number of subjects, by 100 nodes
    cluster_idxs, cluster_names = load_labeled_clusters(base_dir, data_dir, model_name, subjects, n_clusters)
    (orig_cluster_profiles, orig_subject_profiles) = get_cluster_subject_afq_profiles(
        base_dir, data_dir, subjects, bundle_name, cluster_idxs, cluster_names, n_clusters
    )

    new_cluster_profiles = {}
    new_subject_profiles = {}

    # WARNING: each cluster may have different number of profiles (N profiles, 100 nodes)
    # see the note in pydoc
    for target in subjects:
        cluster_idxs, cluster_names = load_relabeled_clusters(base_dir, data_dir, model_name, subjects, n_clusters, target)
        (new_cluster_profiles[target], new_subject_profiles[target]) = get_cluster_subject_afq_profiles(
            base_dir, data_dir, subjects, bundle_name, cluster_idxs, cluster_names, n_clusters
        )
        
    return (bundle_profiles, orig_cluster_profiles, orig_subject_profiles, new_cluster_profiles, new_subject_profiles)


def calculate_ratios(base_dir, data_dir, model_name, subjects, bundle_name, n_clusters):
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
    ) = get_profiles(base_dir, data_dir, model_name, subjects, bundle_name, n_clusters)

    ###################################
    # baseline comparison -- calculate ratio for bundle
    # to determine whether clustering beneficial
    # NOTE optional -- not used in determining consensus subject

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

    ###################################
    # baseline comparison -- calculate the original cluster ratios
    # to determine whether relabeling beneficial
    # NOTE optional -- not used in determining consensus subject

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


def find_consensus_subject(base_dir, data_dir, model_name, subjects, bundle_name, n_clusters, algorithm=Algorithm.MUNKRES):
    """
    TODO need to document, but more importantly need to select and return consensus subject
    """
    import time
    from os.path import join
    
    # run step 0
    logger.log(logging.INFO, 'loading MNI clusters')
    _tic = time.perf_counter()
    tractograms, tractograms_filenames = load_MNI_cluster_tractograms(base_dir, data_dir, model_name, subjects, bundle_name, n_clusters)
    _toc = time.perf_counter()
    logger.log(logging.DEBUG, f'loading MNI clusters {_toc - _tic:0.4f} seconds')

    # run step 1
    for subject in subjects:
        logger.log(logging.INFO, f'matching clusters to subject {subject}')
        _tic = time.perf_counter()
        match_clusters(base_dir, data_dir, model_name, subjects, tractograms, tractograms_filenames, subject, n_clusters, algorithm)
        _toc = time.perf_counter()
        logger.log(logging.DEBUG, f'matching clusters to subject {subject} {_toc - _tic:0.4f} seconds')

    # run step 2
    logger.log(logging.INFO, 'calculating ratios')
    _tic = time.perf_counter()
    _, _, new_ratios = calculate_ratios(base_dir, data_dir, model_name, subjects, bundle_name, n_clusters)
    _toc = time.perf_counter()
    logger.log(logging.DEBUG, f'calculating ratios {_toc - _tic:0.4f} seconds')

    # choose the minimum profile ratio as consensus subject
    _, idx = min((val, idx) for (idx, val) in enumerate(new_ratios))

    # NOTE one issue to consider is how many bundles is available in retest
    # if the subject has different number of test-retest clusters, probably not
    # ideal choice

    logger.log(logging.INFO, f'consensus subject {subjects[idx]}')

    return subjects[idx]


###############################################################################
###############################################################################
# Step 4:
# • Once identified 'consensus subject', repeat for each session 
# NOTE: Ensure using same target subject across sessions: no peaking!
# • align labels across sessions as before.
###############################################################################
###############################################################################

def load_consensus_subject(base_dir, data_dir, model_name, subjects, bundle_name, subject, n_clusters, algorithm=Algorithm.MUNKRES):
    tractograms, tractograms_filenames = load_MNI_cluster_tractograms(base_dir, data_dir, model_name, subjects, bundle_name, n_clusters)
    match_clusters(base_dir, data_dir, model_name, subjects, tractograms, tractograms_filenames, subject, n_clusters, algorithm)

###############################################################################
###############################################################################
# Main
###############################################################################
###############################################################################

# TODO: make clean function to remove files created by this script

# Now that have identified "consensus" subject (target)

# match cluster retest
def match_retest_clusters(base_dir, model_name, subjects, bundle_name, consensus, n_clusters, algorithm=Algorithm.MUNKRES):
    data_dir = 'HCP_Retest'
    tractograms, tractograms_filenames = load_MNI_cluster_tractograms(base_dir, data_dir, model_name, subjects, bundle_name, n_clusters)
    match_clusters(base_dir, data_dir, model_name, subjects, tractograms, tractograms_filenames, consensus, n_clusters, algorithm)

# visualize the clusters profiles across test-retest
def get_cluster_afq_profiles(base_dir, data_dirs, model_name, subjects, bundle_name, n_clusters, target):
    """
    see visualizations.get_cluster_afq_profiles
    """
    from os.path import join
    from dipy.stats.analysis import afq_profile, gaussian_weights

    cluster_afq_profiles = {}

    for subject in subjects:
        cluster_afq_profiles[subject] = {}
        for data_dir in data_dirs:
            cluster_afq_profiles[subject][data_dir] = {}
            fa_scalar_data = load_fa_scalar_data(base_dir, data_dir, subjects)
            bundle_tractograms = load_bundle_tractograms(base_dir, data_dir, subjects, bundle_name)
            cluster_idxs, cluster_names = load_relabeled_clusters(base_dir, data_dir, model_name, subjects, n_clusters, target)
            ii = 0
            for cluster_name, cluster_idx in zip(cluster_names[subject], cluster_idxs[subject]):
                cluster_afq_profiles[subject][data_dir][ii] = afq_profile(
                    fa_scalar_data[subject],
                    bundle_tractograms[subject].streamlines[cluster_idx],
                    bundle_tractograms[subject].affine,
                    weights=gaussian_weights(bundle_tractograms[subject].streamlines[cluster_idx])
                )
                ii += 1

    return cluster_afq_profiles


def cluster_reliability(base_dir, data_dirs, model_name, subjects, bundle_name, n_clusters, target):
    """
    plot the test and retest afq cluster profiles with confidence intervals
    """
    import visualizations as viz

    cluster_afq_profiles = get_cluster_afq_profiles(base_dir, data_dirs, model_name, subjects, bundle_name, n_clusters, target)
    model_names, _, _, cluster_names, _ = viz.load_clusters(base_dir, bundle_name)

    # rewrite model_names and cluster names
    for subject in subjects:
        for data_dir in data_dirs:
            model_names[subject][data_dir] = ['mase_fa_r2_is_mdf']

    for subject in subjects:
        for data_dir in data_dirs:
            _, _cluster_names = load_relabeled_clusters(base_dir, data_dir, model_name, subjects, n_clusters, target)
            cluster_names[subject][data_dir] = [_cluster_names[subject]]

    viz.plot_cluster_reliability(base_dir, bundle_name, 'fa', cluster_afq_profiles, model_names, cluster_names)

"""
WIP
"""
# def population_reliability(base_dir, data_dirs, model_name, subjects, bundle_name, target):
#     """
#     Plot the bar chart

#     TODO NOT FINISHED PORTING from visualizations.py
#     """
#     import visualizations as viz
#     from os.path import join
#     import numpy as np

#     fa_scalar_data = viz.load_fa_scalar_data(base_dir)
#     md_scalar_data = viz.load_md_scalar_data(base_dir)
#     tractograms = viz.load_tractograms(base_dir, bundle_name)
#     model_names, _, cluster_idxs, cluster_names, _ = viz.load_clusters(base_dir, bundle_name)

#     for subject in subjects:
#         for data_dir in data_dirs:
#             model_names[subject][data_dir] = ['mase_fa_r2_is_mdf']

#     for subject in subjects:
#         for data_dir in data_dirs:
#             _cluster_idxs, _cluster_names = load_relabeled_clusters(base_dir, data_dir, model_name, subjects, target)
#             cluster_idxs[subject][data_dir] = [_cluster_idxs[subject]]
#             cluster_names[subject][data_dir] = [_cluster_names[subject]]

#     cluster_afq_profiles = get_cluster_afq_profiles(base_dir, data_dirs, model_name, subjects, bundle_name, target)

#     bundle_dice_coef = viz.get_bundle_dice_coefficients(base_dir, tractograms)

#     # TODO calculate cluster_dice_coef
#     # cluster_dice_coef = viz.get_cluster_dice_coefficients(base_dir, bundle_name, model_names, cluster_names)

#     bundle_profile_fa_r2 = viz.get_bundle_reliability(base_dir, 'fa', fa_scalar_data, tractograms)

#     cluster_profile_fa_r2 = viz.get_cluster_reliability(base_dir, bundle_name, cluster_afq_profiles, cluster_names)

#     # ignore md for now
#     bundle_profile_md_r2 = {}
#     cluster_profile_md_r2 = {}

#     for subject in subjects:
#         bundle_profile_md_r2[subject] = 0.0
#         cluster_profile_md_r2[subject] = np.zeros((3,3))

#     viz.population_visualizations(base_dir, bundle_name, bundle_dice_coef, cluster_dice_coef, bundle_profile_fa_r2, cluster_profile_fa_r2, bundle_profile_md_r2, cluster_profile_md_r2, model_names, cluster_names)