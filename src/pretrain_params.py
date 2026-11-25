import datetime
import os


clip_dir = '/home/ankit.dhurve/datasets/DFEW/clip_224x224_16f'

training_label = '/home/ankit.dhurve/datasets/DFEW/train_labels/train(single-labeled)/set_2.csv'
test_label = '/home/ankit.dhurve/datasets/DFEW/test_labels/test(single-labeled)/set_2.csv'

num_frames = 16

num_classes = 7

batch_size = 8

num_workers = 8

num_epochs = 60

learning_rate = 1e-5

image_size = 224 

dfew_emotion_classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

dfew_emotion_classes_dict = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3, 'Sad': 4, 'Surprise': 5, 'Neutral': 6}

dfew_emotion_classes_dict_reverse = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

gpu_ids = [0, 1]

pretrain_weights_dir = '/home/ankit.dhurve/ppfer-main (2)/ppfer-main/outputs'



# model saving path
now = datetime.datetime.now()
time_str = 'pretrain-'+now.strftime("[%Y-%m-%d]-[%H:%M]")

# The output path is the folder where the model will be saved
output_path='../outputs' 
pretriain_weights_dir = '../outputs'
output_path_time = os.path.join(output_path,time_str)
if not os.path.exists(output_path_time):
    os.mkdir(output_path_time)
# traning intermediate results
tensorboard_log_path = os.path.join(output_path_time, 'tensorboard_log')
log_path = os.path.join(output_path_time, 'run_logs.log')
