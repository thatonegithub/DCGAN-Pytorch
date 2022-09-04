# General options

# images will be resized to this value
op_image_size = 64

# number of channels in the images (3 for RGB)
op_nc = 1

# batch size for both networks
op_batch_size = 64

# number of GPUs to use (pass 0 for CPU)
op_gpu = 1

# maximum convolution channels to incorporate
op_conv_channels = 256

# latent space size (100 in the paper)
op_nz = 64

# set 'True' to enable bias for both networks
op_enable_bias = False

# fake & real lable values
op_fake_lables = 0.0
op_real_lables = 1.0

# directory to save the state of networks & optimizers
op_state_dir = 'state'

# name of the save file
op_state_file = 'state.tar'




# Training options

# where to locate the training data
op_train_data_path = 'data'

# pytorch cpu worker count
op_train_workers = 2

# how many times to enumerate the dataset
op_train_num_epochs = 4

# Adam optimizer configuration for beta1 (set according to the paper) beta2 is set to 0.999 in the train.py file
op_train_beta1 = 0.5

# whether to resume optimization from a saved state or start from scratch
op_train_load_state = True

# learning rate
op_train_learning_rate = 0.0002

# how many steps until a checkpoint is reached
op_train_save_steps = 250



# Making samples

# where to put the generated samples
op_ms_output_dir = 'output'
# what device to use for generating samples
op_ms_device = 'cpu'

# name & lable for output sample image
op_ms_output_name = 'sample.jpg'
op_ms_output_title = 'Synthesized Images'

# video settings for interpolating the latent space
op_ms_video_fps = 24
op_ms_video_length = 10 # in seconds
op_ms_video_gpu = True
op_ms_video_steps = op_ms_video_fps
op_ms_video_title = 'Latent Space Interpolation'
op_ms_video_name = 'sample.mp4'
