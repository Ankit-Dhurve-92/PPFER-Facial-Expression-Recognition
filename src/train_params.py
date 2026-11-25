import os
import datetime

BASE_DIR = "/home/ankit.dhurve/ppfer-main (2)/ppfer-main"
DATA_DIR = "/home/ankit.dhurve/datasets/CREMA-D"

# model pretrain weights
unet_pretrain_weight = os.path.join(BASE_DIR, "outputs/pretrain-[2026-11-22]-[13:23]/f_a_init_weight.pth")
resnet50_pretrain_weight = os.path.join(BASE_DIR, "pretrained_weights/c_hpr_identity.pth")
c_lpr_pretrain_weight = os.path.join(BASE_DIR, "pretrained_weights/c_lpr_emotion.pth")

# log dir
now = datetime.datetime.now()
time_str = 'train-' + now.strftime("[%Y-%m-%d]-[%H:%M]")
output_path = os.path.join(BASE_DIR, "outputs")
os.makedirs(output_path, exist_ok=True)
output_path_time = os.path.join(output_path, time_str)
os.makedirs(output_path_time, exist_ok=True)

# training intermediate results
tensorboard_log_path = os.path.join(output_path_time, 'tensorboard_log')
log_path = os.path.join(output_path_time, 'run_logs.log')

# data loader
s1_video_clip_dir = os.path.join(DATA_DIR, "clips")
s2_video_clip_dir = os.path.join(DATA_DIR, "s2_clips")
s3_video_clip_dir = os.path.join(DATA_DIR, "s3_clips")
train_identity_file = os.path.join(DATA_DIR, "train_identity.csv")
train_emotion_file = os.path.join(DATA_DIR, "train_emotion.csv")
test_identity_file = os.path.join(DATA_DIR, "test_identity.csv")
test_emotion_file = os.path.join(DATA_DIR, "test_emotion.csv")
num_frames = 16
image_size = 256

# training
num_epochs = 15
batch_size = 6
num_workers = 8
learning_rate = 1e-4
gpu_ids = [0, 1]

# the paths of best models at stage 1
best_f_hpr = os.path.join(output_path_time, 'f_hpr_best.pth')
best_f_lpr = os.path.join(output_path_time, 'f_lpr_best.pth')

# paths for saving stage 1 intermediate results
training_batch_tensor_path = ""
validation_batch_tensor_path = ""

# stage 2
uvit_256_weight = os.path.join(BASE_DIR, "pretrained_weights/imagenet256_uvit_large.pth")
dfew_image_fer_weight = os.path.join(BASE_DIR, "pretrained_weights/cremad_fer_controller.pth")

# s2 training
s2_learning_rate = 1e-4
s2_batch_size = 6
