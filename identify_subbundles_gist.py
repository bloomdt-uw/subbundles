"""
code snippets (gists) to run in python intepreter

after generating cluster model studys published on aws s3 repository
and having identifed consensus subject create centroids for 
VisualizeCentroids.ipynb

TODO: incorporate into notebook
"""

from identify_subbundles import *

subj_centroids = prealignment_centroids(BASE_DIR, 'HCP_1200', MODEL_NAME, SUBJECTS, 'SLF')
mni_centroids = move_centriods_to_MNI('HCP_1200', SUBJECTS, subj_centroids)
target = '200614'
sources = SUBJECTS[:]
sources.remove(target)
match_clusters_by_centroid_MDF(BASE_DIR, 'HCP_1200', MODEL_NAME, mni_centroids, target, sources)