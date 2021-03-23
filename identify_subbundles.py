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
        afd.fetch_hcp([subject], study=data_dir)

    # ~/AFQ_data/HCP_1200/derivatives/dmriprep/sub-125525/ses-01/dwi/sub-125525_dwi.nii.gz
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
    mapping = reg.read_mapping(mapping_file, dwi_img, MNI_T2_img) # both the forward and backward transformations in MNI space

    ## validate mapping
    # np.shape(mapping.forward) -- (193, 229, 193, 3)
    # np.shape(mapping.backward) -- (193, 229, 193, 3)

    ###############################################################################
    # move subject streamlines into MNI space

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


def load_MNI_clusters(base_dir, data_dir, model_name, subjects, bundle_name):
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
    
    NOTE: Usage of global variables: K

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

    Returns
    -------
    tractograms : dict
        dict of list of StatefulTractograms. each StatefulTractorgram represents
        the subjects cluster in MNI space. the list contains a StatefulTractogram
        for each cluster assigned by the clustering model. the dict stores each
        list by the subject.
    """
    from os.path import join, exists
    from dipy.io.streamline import load_tractogram, save_tractogram

    tractograms = {}

    for subject in subjects:
        tractograms[subject] = []
        subbundle_base_dir = join(base_dir, subject, data_dir)
        
        for cluster_id in range(K):
            MNI_tractogram_file = join(subbundle_base_dir, f'{subject}_{bundle_name}_{cluster_id}_MNI.trk')

            # load subject clusters into MNI space
            if not exists(MNI_tractogram_file):
                tractogram_name = f'{model_name}_cluster_{cluster_id}_clean.trk'
                tractogram_file = join(subbundle_base_dir, tractogram_name)
                
                if not exists(tractogram_file):
                    print(f'{subject} {bundle_name} {cluster_id} not found')
                    continue

                tractogram = load_tractogram(tractogram_file, 'same')

                sft = move_tractogram_to_MNI_space(data_dir, subject, tractogram)

                ## Save tractogram to visually inspect in MI Brain
                # NOTE: remember to globally reinitize MI-Brain 
                # (otherwise streamlines may not be loaded correctly)
                # TODO: Ideally want to generate images using AFQ.viz.plotly

                # NOTE: some streamlines appears outside the brain boundary in the MNI template image
                save_tractogram(sft, MNI_tractogram_file, bbox_valid_check=False)
            else:
                sft = load_tractogram(MNI_tractogram_file, 'same')

            tractograms[subject].append(sft)
    
    return tractograms


###############################################################################
###############################################################################
# Step 1:
# • relabel clusters assuming a single fixed target subject
###############################################################################
###############################################################################

def get_density_map(tractogram):
    """
    Take a tractogram and return a binary image of the streamlines,
    these images is used to calculate the dice coefficents to compare
    cluster similiartiy.

    Parameters
    ----------
    tractogram : StatefulTractogram

    Returns
    -------
    density_map_img : Nifti1Image
    """
    import numpy as np
    from dipy.io.utils import create_nifti_header, get_reference_info
    import dipy.tracking.utils as dtu
    import nibabel as nib

    affine, vol_dims, voxel_sizes, voxel_order = get_reference_info(tractogram)
    tractogram_density = dtu.density_map(tractogram.streamlines, np.eye(4), vol_dims)
    # force to unsigned 8-bit; done to reduce the size of the density map image
    tractogram_density = np.uint8(tractogram_density)
    nifti_header = create_nifti_header(affine, vol_dims, voxel_sizes)
    density_map_img = nib.Nifti1Image(tractogram_density, affine, nifti_header)

    # TODO could cache/save the density map image

    return density_map_img


def get_dice_coefficients(source_sft, target_sft):
    """
    Given two tractograms determine the similary by calculating the weighted
    dice coefficient. Assumes source and target are colocated in same space.

    Parameters
    ----------
    source_sft : StatefulTractogram

    target_sft : StatefulTractogram

    Returns
    -------
    dice_coeff : int
        weighted dice coefficient
    """
    from AFQ.utils.volume import dice_coeff

    source_sft.to_vox()
    source_map = get_density_map(source_sft)

    target_sft.to_vox()
    target_map = get_density_map(target_sft)

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
    
    label_map = dict(zip(np.arange(len(new_labels)), new_labels))

    relabeled_clusters = np.copy(labeled_clusters)

    for original_label, new_label in label_map.items():
        relabeled_clusters[labeled_clusters == original_label] = new_label

    return relabeled_clusters


def match_clusters_by_maximum_dice(base_dir, data_dir, model_name, tractograms, target, sources):
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

    target_subbundle_base_dir = join(base_dir, target, data_dir)

    # use remaining subjects for pairwise comparison
    for source in sources:
        adjacency_block_file = join(target_subbundle_base_dir, f'{target}_{source}_adjacency_block.npy')
        cluster_labels_file = join(target_subbundle_base_dir, f'{target}_{source}_cluster_labels.npy')

        if not exists(cluster_labels_file):
            pairwise_dice = np.array([])
            
            # compare source and target clusters using weighted dice
            for (source_tractogram, target_tractogram) in itertools.product(tractograms[source], tractograms[target]):
                pairwise_dice = np.append(pairwise_dice, get_dice_coefficients(source_tractogram, target_tractogram))

            pairwise_shape = (len(tractograms[source]), len(tractograms[target]))
            block = pairwise_dice.reshape(pairwise_shape)
            # print(source, target)
            # print(block)

            # NOTE: could put the block into a dict with key source:target
            # then when looking for subject pairs
            # if key doesn't exist, then 
            # reverse subject order in pair and if exists then 
            # need to transpose block

            # save block
            np.save(adjacency_block_file, block)
            
            # take the source and move label to correspond to target cluster with highest overlap
            # relabel clusters: i -> ids[i]
            # [1 0 0] => 0 -> 1, 1 -> 0, 2 -> 0 # subject 1 to subject 0
            # [0 0 0] => 0 -> 0, 1 -> 0, 2 -> 0 # subject 2 to subject 0
            # [1 2 0] => 1 -> 0, 1 -> 2, 2 -> 0 # subject 2 to subject 1
            target_labels = np.argmax(block, axis=1)
            # print(target_labels)

            # save relabeling
            np.save(cluster_labels_file, target_labels)

            # print(block.T)
            # # [2 0] => 0 -> 2, 1 -> 0 # subject 0 to subject 1
            # # [0 2] => 0 -> 0, 1 -> 2 # subject 0 to subject 2
            # # [0 1 1] => 0 -> 0, 1 -> 1, 2 -> 1 # subject 1 to subject 2
            # print(np.argmax(block.T, axis=1))
            # print(np.argmax(block, axis=0)) # alternatively
        else:
            target_labels = np.load(cluster_labels_file)

        # load original source cluster labels and create new source cluter label file reindexed with target labels
        # this can be used with the bundle tractogram to load clusters
        source_subbundle_base_dir = join(base_dir, source, data_dir)
        original_cluster_labels = np.load(join(source_subbundle_base_dir, f'{model_name}_idx.npy'))
        new_cluster_labels = relabel_clusters(original_cluster_labels, target_labels)
        np.save(join(source_subbundle_base_dir, f'{target}_idx.npy'), new_cluster_labels)

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

def match_clusters_by_best_remaining_dice(base_dir, data_dir, model_name, tractograms, target, sources):
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

    target_subbundle_base_dir = join(base_dir, target, data_dir)

    # use remaining subjects for pairwise comparison
    for source in sources:
        adjacency_block_file = join(target_subbundle_base_dir, f'{target}_{source}_adjacency_block.npy')
        munkres_cluster_labels_file = join(target_subbundle_base_dir, f'{target}_{source}_munkres_cluster_labels.npy')

        if not exists(munkres_cluster_labels_file):
            pairwise_dice = np.array([])
            
            # compare source and target clusters using weighted dice
            for (source_tractogram, target_tractogram) in itertools.product(tractograms[source], tractograms[target]):
                pairwise_dice = np.append(pairwise_dice, get_dice_coefficients(source_tractogram, target_tractogram))

            pairwise_shape = (len(tractograms[source]), len(tractograms[target]))
            block = pairwise_dice.reshape(pairwise_shape)
            # print(source, target)
            # print(block)

            # NOTE: could put the block into a dict with key source:target
            # then when looking for subject pairs
            # if key doesn't exist, then 
            # reverse subject order in pair and if exists then 
            # need to transpose block

            # save block
            np.save(adjacency_block_file, block)

            target_labels = maximize_trace(block)
            # print(target_labels)
            np.save(munkres_cluster_labels_file, target_labels)

        else:
            target_labels = np.load(munkres_cluster_labels_file, allow_pickle=True)

        # load original source cluster labels and create new source cluter label file reindexed with target labels
        # this can be used with the bundle tractogram to load clusters
        source_subbundle_base_dir = join(base_dir, source, data_dir)
        original_cluster_labels = np.load(join(source_subbundle_base_dir, f'{model_name}_idx.npy'))
        new_cluster_labels = relabel_clusters(original_cluster_labels, target_labels)
        np.save(join(source_subbundle_base_dir, f'{target}_munkres_idx.npy'), new_cluster_labels)


###############################################################################
# Since relabeling using weighted DICE coefficients is underperforming, 
# try using centriod MDF
###############################################################################

def prealignment_centroids(base_dir, data_dir, model_name, subjects, bundle_name):
    """
    calculate the centriod of each cluster for each subject

    Parameters
    ----------
    base_dir : string
    data_dir : string
    model_name : string
    subjects : array
    bundle_name : string

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
        subbundle_base_dir = join(base_dir, subject, data_dir)
        
        for cluster_id in range(K):
            tractogram_name = f'{model_name}_cluster_{cluster_id}_clean.trk'
            tractogram_file = join(subbundle_base_dir, tractogram_name)
            
            if not exists(tractogram_file):
                print(f'{subject} {bundle_name} {cluster_id} not found')
                continue

            tractogram = load_tractogram(tractogram_file, 'same')

            centroid = np.mean(set_number_of_points(tractogram.streamlines, n_points), axis=0)
            
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
    
    bundle_tractograms = load_tractograms(base_dir, data_dir, subjects, bundle_name)

    for subject in subjects:
        centroids[subject] = []
        for cluster_name, cluster_idx in zip(cluster_names[subject], cluster_idxs[subject]):
            centroid = np.mean(set_number_of_points(bundle_tractograms[subject].streamlines[cluster_idx], n_points), axis=0)
            
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
            mni_centriods[subject].append(move_tractogram_to_MNI_space(data_dir, subject, centroid))
    
    return mni_centriods

def match_clusters_by_centroid_MDF(base_dir, data_dir, model_name, centriods, target, sources):
    """
    """
    import numpy as np
    import itertools
    from dipy.tracking.streamline import bundles_distances_mdf

    # For each subject pair compute the adjacency matrix
    for source in sources:
        pairwise_mdf = np.array([])
            
        # calculate distance between source and target centroids
        for (source_centriod, target_centroid) in itertools.product(centriods[source], centriods[target]):
            mdf = bundles_distances_mdf(source_centriod.streamlines, target_centroid.streamlines)
            pairwise_mdf = np.append(pairwise_mdf, mdf)

        pairwise_shape = (len(centriods[source]), len(centriods[target]))
        block = pairwise_mdf.reshape(pairwise_shape)
        # print(target, source, block)

        # TODO want to minimize the trace 
        target_labels = np.argmax(block, axis=1)

        source_subbundle_base_dir = join(base_dir, source, data_dir)
        original_cluster_labels = np.load(join(source_subbundle_base_dir, f'{model_name}_idx.npy'))
        new_cluster_labels = relabel_clusters(original_cluster_labels, target_labels)
        print(target, source, new_cluster_labels)
        np.save(join(source_subbundle_base_dir, f'{target}_mdf_idx.npy'), new_cluster_labels)

###############################################################################

def match_clusters(base_dir, data_dir, model_name, subjects, tractograms, target):
    """
    Run matching algorithms for `target` on remaining `subjects` in dataset.

    New cluster labels are saved to disk. See `match_clusters_by_maximum_dice` 
    and `match_clusters_by_best_remaining_dice` for details.
    """
    sources = subjects[:]
    sources.remove(target)
    # print('matching by maximum dice')
    # match_clusters_by_maximum_dice(base_dir, data_dir, model_name, tractograms, target, sources)
    print('matching by munkres')
    match_clusters_by_best_remaining_dice(base_dir, data_dir, model_name, tractograms, target, sources)


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

        scalar_data[subject] = nib.load(scalar_filename).get_fdata()

    return scalar_data


def load_tractograms(base_dir, data_dir, subjects, bundle_name, csd=True):
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
        
        tractogram = load_tractogram(tractogram_filename, 'same')
        tractograms[subject] = tractogram

    return tractograms


def load_labeled_clusters(base_dir, data_dir, model_name, subjects):
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
        cluster_file = join(base_dir, subject, data_dir, f'{model_name}_idx.npy')

        cluster_labels = np.load(cluster_file)
        cluster_names[subject] = np.unique(cluster_labels)
        cluster_idxs[subject] = np.array([np.where(cluster_labels == i)[0] for i in np.unique(cluster_labels)])
    
    return (cluster_idxs, cluster_names)


def load_relabeled_clusters(base_dir, data_dir, model_name, subjects, target, algorithm=None):
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

        cluster_file = join(base_dir, subject, data_dir, cluster_filename)

        cluster_labels = np.load(cluster_file)
        cluster_names[subject] = np.unique(cluster_labels)
        cluster_idxs[subject] = np.array([np.where(cluster_labels == i)[0] for i in np.unique(cluster_labels)])
    
    return (cluster_idxs, cluster_names)


def get_cluster_subject_afq_profiles(base_dir, data_dir, subjects, bundle_name, cluster_idxs, cluster_names):
    """
    Calculate all afq profiles for each cluster. Where the cluster labels 
    originate from the cluster model `model_name`.

    Returns a two dictionaries
    
    one where the key is the cluster label and each value contains an array of
    arrays with each subjects weighted afq profile. used to calculate within
    cluster variance.

    one where the key is the subject id and each value contains an array of
    arrays with each clusters weighted afq profile. used to calculate across
    cluster variance.
    """
    # subbundle profiles from original custer labels
    from os.path import join
    from dipy.stats.analysis import afq_profile, gaussian_weights

    fa_scalar_data = load_fa_scalar_data(base_dir, data_dir, subjects)

    # using tractograms - which have been transformed into MNI space is not going to work...
    # therefore need to load the cluster ids
    bundle_tractograms = load_tractograms(base_dir, data_dir, subjects, bundle_name)
    
    # TODO: HARD CODING LABELS!
    cluster_profiles = {}
    cluster_profiles[0] = []
    cluster_profiles[1] = []
    cluster_profiles[2] = []

    subject_profiles = {}

    for subject in subjects:
        subject_profiles[subject] = []
        for cluster_name, cluster_idx in zip(cluster_names[subject], cluster_idxs[subject]):
            # print(subject, cluster_name)
            # print(len(bundle_tractograms[subject].streamlines), len(bundle_tractograms[subject].streamlines[cluster_idx]))
            
            # TODO: save to disk calculating these profiles is time intenstive
            profile = afq_profile(
                fa_scalar_data[subject],
                bundle_tractograms[subject].streamlines[cluster_idx],
                bundle_tractograms[subject].affine,
                weights=gaussian_weights(bundle_tractograms[subject].streamlines[cluster_idx])
            )

            cluster_profiles[cluster_name].append(profile)
            subject_profiles[subject].append(profile)

    return (cluster_profiles, subject_profiles)


def get_bundle_afq_profiles(base_dir, data_dir, subjects, bundle_name):
    """
    Calculate the weighted afq profile for each subjects bundle.

    Return array of arrays with each subjects weighted afq profile.
    """
    from os.path import join
    from dipy.stats.analysis import afq_profile, gaussian_weights

    fa_scalar_data = load_fa_scalar_data(base_dir, data_dir, subjects)
    bundle_tractograms = load_tractograms(base_dir, data_dir, subjects, bundle_name)

    profiles = []

    for subject in subjects:
        profiles.append(
            afq_profile(
                fa_scalar_data[subject],
                bundle_tractograms[subject].streamlines,
                bundle_tractograms[subject].affine,
                weights=gaussian_weights(bundle_tractograms[subject].streamlines)
            )
        )

    return profiles


def calculate_variance(base_dir, data_dir, model_name, subjects, bundle_name):
    # TODO split into profile retrival ans storage and calculating metrics
    """
    Really this is calculating the mean of the standard deviation for each
    of the following FA afq profiles: bundle, original model clusters, new
    clusters based on target subject. This will allow us to assess cluster
    performance

    apply otsu's critria 
    - minimize variance within subbundles
    - maximize variance across subbundles

    returns the afq_profiles for bundle, original, and new as a convenience

    NOTE: much of the downstream functions depend on assumptions and global variables
    """
    import numpy as np

    bundle_profiles = get_bundle_afq_profiles(base_dir, data_dir, subjects, bundle_name)
    
    # calculate the mean of standard deviation of the fa profile for each 
    # subjects bundle. 
    # NOTE: that bundle_profiles consists of (44 subjects, 100 nodes)
    bundle_var = np.mean(np.std(np.array(bundle_profiles), axis=0))
    subject_var = np.mean(np.std(np.array(bundle_profiles), axis=1))
    
    print("bundle", bundle_name)
    print(bundle_var, subject_var)
    print('ratio', bundle_var/subject_var)

    ###################################

    # dictionary of K clusters with N profiles, where N <= number of subjects, by 100 nodes
    cluster_idxs, cluster_names = load_labeled_clusters(base_dir, data_dir, model_name, subjects)
    (orig_cluster_profiles, orig_subject_profiles) = get_cluster_subject_afq_profiles(
        base_dir, data_dir, subjects, bundle_name, cluster_idxs, cluster_names
    )

    # within cluster variation
    # calculate the mean of standard deviation of the fa profile for each subjects
    # cluster. 
    orig_cluster_0_var = np.mean(np.std(np.array(orig_cluster_profiles[0]), axis=0))
    orig_cluster_1_var = np.mean(np.std(np.array(orig_cluster_profiles[1]), axis=0))
    orig_cluster_2_var = np.mean(np.std(np.array(orig_cluster_profiles[2]), axis=0))

    # take the mean of means across all clusters
    total_orig_cluster_var = np.nanmean(
        [
            orig_cluster_0_var,
            orig_cluster_1_var,
            orig_cluster_2_var
        ]
    )

    # within subject variation
    total_orig_subject_var = np.mean(np.array([np.mean(np.std(np.array(orig_subject_profiles[subject]))) for subject in subjects]))
    

    print("original clusters")
    print(total_orig_cluster_var, total_orig_subject_var)
    print('ratio', total_orig_cluster_var/total_orig_subject_var)

    ###################################

    new_cluster_0_vars = []
    new_cluster_1_vars = []
    new_cluster_2_vars = []
    total_new_cluster_vars = []

    for target in subjects:
        # NOTE: Recall
        # 1) Depending on number of clusters found, some subjects have two
        #    clusters and others have three, and 
        # 2) The choice of relabeling algorithm the number of subjects clusters may
        #    merge (as in the non-Munkres case).
        # 
        # Thus each cluster may have different number of profiles 
        # (N profiles, 100 nodes)
        cluster_idxs, cluster_names = load_relabeled_clusters(base_dir, data_dir, model_name, subjects, target)
        (new_cluster_profiles, new_subject_profiles) = get_cluster_subject_afq_profiles(
            base_dir, data_dir, subjects, bundle_name, cluster_idxs, cluster_names
        )

        # within cluster variation
        # calculate the mean of standard deviation of the fa profile for each subjects
        # cluster. 
        new_cluster_0_var = np.mean(np.std(np.array(new_cluster_profiles[0]), axis=0))
        new_cluster_0_vars.append(new_cluster_0_var)
        new_cluster_1_var = np.mean(np.std(np.array(new_cluster_profiles[1]), axis=0))
        new_cluster_1_vars.append(new_cluster_1_var)
        new_cluster_2_var = np.mean(np.std(np.array(new_cluster_profiles[2]), axis=0))
        new_cluster_2_vars.append(new_cluster_2_var)
        # take the mean of means across all clusters
        total_new_cluster_var = np.nanmean(
            [
                new_cluster_0_var,
                new_cluster_1_var,
                new_cluster_2_var
            ]
        )
        total_new_cluster_vars.append(total_new_cluster_var)

        # within subject variation
        total_new_subject_var = np.mean(np.array([np.mean(np.std(np.array(new_subject_profiles[subject]))) for subject in subjects]))

        print("munkres clusters based on target ", target)
        print(total_new_cluster_var, total_new_subject_var)
        # TODO want to select minimum
        print("ratio", total_new_cluster_var/total_new_subject_var)

    print('min/max total var', min(total_new_cluster_vars), max(total_new_cluster_vars))
    print('min/max total var', min(new_cluster_0_vars), max(new_cluster_0_vars))
    print('min/max total var', min(new_cluster_1_vars), max(new_cluster_1_vars))
    print('min/max total var', min(new_cluster_2_vars), max(new_cluster_2_vars))
    # calculate the ratio a pseudo F-score
    # ratio = total_var/bundle_var

    return (bundle_profiles, orig_cluster_profiles, orig_subject_profiles, new_cluster_profiles, new_subject_profiles)


# TODO now that have metric choose subject with minimal ratio as consensus
# NOTE one issue to consider is how many bundles is available in retest
# if the subject has different number of test-retest clusters, probably not
# ideal choice

###############################################################################
###############################################################################
# Step 3:
# • iterate changing target for each subject
# • choose the label with lowest variance in FA profiles call this the
#   "true" cluster label - i.e. subbundle
###############################################################################
###############################################################################

# TODO choose min from test set where all clusters.
# manually selected at the moment
# OTSU's criteria as ratio

def find_consensus_subject(base_dir, data_dir, model_name, subjects, bundle_name):
    """
    TODO need to document, but more importantly need to select and return consensus subject
    """
    import time
    from os.path import join
    
    # run step 0
    print('loading MNI clusters')
    _tic = time.perf_counter()
    tractograms = load_MNI_clusters(base_dir, data_dir, model_name, subjects, bundle_name)
    _toc = time.perf_counter()
    print(f'loading MNI clusters {_toc - _tic:0.4f} seconds')

    # run step 1
    for subject in subjects:
        print(f'matching clusters to subject {subject}')
        _tic = time.perf_counter()
        match_clusters(base_dir, data_dir, model_name, subjects, tractograms, subject)
        _toc = time.perf_counter()
        print(f'matching clusters to subject {subject} {_toc - _tic:0.4f} seconds')

    # run step 2
    print('calculating variance')
    _tic = time.perf_counter()
    calculate_variance(base_dir, data_dir, model_name, subjects, bundle_name)
    _toc = time.perf_counter()
    print(f'calculating variance {_toc - _tic:0.4f} seconds')


###############################################################################
###############################################################################
# Step 4:
# • Once identified 'consensus subject', repeat for each session 
# NOTE: Ensure using same target subject across sessions: no peaking!
# • align labels across sessions as before.
###############################################################################
###############################################################################

def load_consensus_subject(base_dir, data_dir, model_name, subjects, bundle_name, subject):
    tractograms = load_MNI_clusters(base_dir, data_dir, model_name, subjects, bundle_name)
    match_clusters(base_dir, data_dir, model_name, subjects, tractograms, subject)

###############################################################################
###############################################################################
# Main
###############################################################################
###############################################################################

# TODO: make clean function to remove files created by this script

###############################################################################
# constants
###############################################################################

###############################################################################
# NOTE: begin small; compare three subjects and use majority vote to relabel cluster
# These subjects were initailly chosen randomly from the 44 subjects in test-restest
# and their data has been downloaded locally

# subjects = ['125525', '175439', '562345']
SUBJECTS = [
    '103818', '105923', '111312', '114823', '115320',
    '122317', '125525', '130518', '135528', '137128',
    '139839', '143325', '144226', '146129', '149337',
    '149741', '151526', '158035', '169343', '172332',
    '175439', '177746', '185442', '187547', '192439',
    '194140', '195041', '200109', '200614', '204521',
    '250427', '287248', '341834', '433839', '562345',
    '599671', '601127', '627549', '660951', # '662551', 
    '783462', '859671', '861456', '877168', '917255'
]

EXPERIMENT_NAME = 'HCP_test_retest_MASE_CSD'
MODEL_NAME = 'mase_fa_r2_is_mdf'
BUNDLE_NAME = 'SLF_L'
from os.path import join
BASE_DIR = join('subbundles', EXPERIMENT_NAME, BUNDLE_NAME)
DATA_DIRS = ['HCP_1200', 'HCP_Retest']
K = 3

###############################################################################
###############################################################################
###############################################################################

# Now that have identifyed "consensus" subject (target)

# match cluster retest
def match_retest_clusters(base_dir, model_name, subjects, bundle_name, consensus):
    data_dir='HCP_Retest'
    tractograms = load_MNI_clusters(base_dir, data_dir, model_name, subjects, bundle_name)
    match_clusters(base_dir, data_dir, model_name, subjects, tractograms, consensus)

# visualize the clusters profiles across test-retest
def get_cluster_afq_profiles(base_dir, data_dirs, model_name, subjects, bundle_name, target):
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
            bundle_tractograms = load_tractograms(base_dir, data_dir, subjects, bundle_name)
            cluster_idxs, cluster_names = load_relabeled_clusters(base_dir, data_dir, model_name, subjects, target)
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


def cluster_reliability(base_dir, data_dirs, model_name, subjects, bundle_name, target):
    """
    plot the test and retest afq cluster profiles with confidence intervals
    """
    import visualizations as viz

    cluster_afq_profiles = get_cluster_afq_profiles(base_dir, data_dirs, model_name, subjects, bundle_name, target)
    model_names, _, _, cluster_names, _ = viz.load_clusters(base_dir, bundle_name)

    # rewrite model_names and cluster names
    for subject in subjects:
        for data_dir in data_dirs:
            model_names[subject][data_dir] = ['mase_fa_r2_is_mdf']

    for subject in subjects:
        for data_dir in data_dirs:
            _, _cluster_names = load_relabeled_clusters(base_dir, data_dir, model_name, subjects, target)
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