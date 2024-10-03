from nilearn.image import load_img, math_img, concat_imgs, iter_img
import os

# Set up paths relative to the script's running directory
data_path = './dataset/'
output_path = './output/'

participant_id = 'sub-0010003'

# Ensure the output directory exists
if not os.path.exists(output_path):
    os.makedirs(output_path)

brain_mask_filename = 'dataset/sub-0010003/sub-0010003_ses-1_task-rest_run-1_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'
preproc_bold_filename = 'dataset/sub-0010003/sub-0010003_ses-1_task-rest_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'

# Load the brain mask and the fMRI data
brain_mask_img = load_img(brain_mask_filename)
subject_img = load_img(preproc_bold_filename)

print('Image loaded, ID: ' + participant_id)
print('Applying brain mask to each time point')

# Apply the brain mask to each time point in the 4D image
masked_imgs = []
for img_3d in iter_img(subject_img):  # Iterate over time points
    masked_img_3d = math_img("img1 * img2", img1=img_3d, img2=brain_mask_img)
    masked_imgs.append(masked_img_3d)

# Concatenate the masked 3D images back into a 4D image
masked_4d_img = concat_imgs(masked_imgs)

# Save the masked 4D image
# masked_4d_img.to_filename(os.path.join(output_path, participant_id + '_masked.nii.gz'))
# print('Masked image saved')


# now we have the masked image, we can use it to do the parcellation using the atlas dataset\aal_resampled_in_mni152_space.nii

# Load the atlas image
atlas_filename = 'dataset/aal_resampled_in_mni152_space.nii' # aal 116
atlas_img = load_img(atlas_filename)

# Apply the atlas to the masked 4D image
from nilearn.input_data import NiftiLabelsMasker

masker = NiftiLabelsMasker(labels_img=atlas_img, standardize=True)

# Fit the masker to the masked 4D image
masker.fit(masked_4d_img)

# Extract the time series from the masked 4D image
time_series = masker.transform(masked_4d_img)

# Save the time series data
import numpy as np

# save as csv, with first row as 0, 1, 2, .... and the content not in scientific notation
output_csv_filename = os.path.join(output_path, participant_id + '_time_series.csv')
np.savetxt(output_csv_filename, time_series, delimiter=",", fmt='%.6f', header=",".join(map(str, range(time_series.shape[1]))), comments='')
print('Time series data saved')


