from dipy.viz import window, actor, colormap as cmap
import nibabel as nib
thresh = 0.5
img = nib.load('cluster_0.nii')
data = img.get_fdata()
data[data>thresh] = 1
data[data<=thresh] = 0
ROI1_actor = actor.contour_from_roi(data, color=(1., 1., 0.), opacity=1)
img = nib.load('cluster_1.nii')
data = img.get_fdata()
data[data>thresh] = 1
data[data<=thresh] = 0
ROI2_actor = actor.contour_from_roi(data, color=(1., 0., 0.), opacity=1)
img = nib.load('cluster_2.nii')
data = img.get_fdata()
data[data>thresh] = 1
data[data<=thresh] = 0
ROI3_actor = actor.contour_from_roi(data, color=(0., 1., 0.), opacity=1)
#vol_actor = actor.slicer(t1_data)
#vol_actor.display(x=40)
#vol_actor2 = vol_actor.copy()
#vol_actor2.display(z=35)
# Add display objects to canvas
scene = window.Scene()
#scene.add(vol_actor)
#scene.add(vol_actor2)
scene.add(ROI1_actor)
scene.add(ROI2_actor)
scene.add(ROI3_actor)
# Save figures
scene.set_camera(position=[-1, 0, 0], focal_point=[0, 0, 0], view_up=[0, 0, 1])
window.show(scene)
window.record(scene, n_frames=1, out_path='rois.png', size=(800, 800))