import ffmpeg
from pydub import AudioSegment
from pydub.silence import split_on_silence
import sys
import os
import numpy as np
import random
def flac_to_wav(filepath, savedir):
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    filename = filepath.replace('.flac', '.wav')
    savefilename = filename.split('/')
    save_dir = savedir + '/' + savefilename[-1]
    print(save_dir)
    cmd = 'ffmpeg -i ' + filepath + ' ' + save_dir
    os.system(cmd)


def flacs_to_wavs(audio_path):
    for root, dirs, files in os.walk(audio_path):
        for dir in dirs:
            save_dir = os.path.join(root, dir).replace('audio_new', 'audio_new_wav')
            for file in os.listdir(os.path.join(root, dir)):
                file_path = os.path.join(root, dir, file)
                if file_path.split('.')[-1]=='flac':
                    flac_to_wav(file_path, save_dir)
################批量flacs文件转wav文件
#audio_path = '/home/gaolili/deepLearning_project/deep-speaker/data/sitw_database.v4/eval/audio_new'
#flacs_to_wavs(audio_path)
###############################
def audios_segments(audio_wav_path, save_dir_root):

    for file in os.listdir(audio_wav_path):
        file_path = os.path.join(audio_wav_path, file)
        save_dir = save_dir_root
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if file_path.split('.')[-1]=='mp3':
            audio_segment = AudioSegment.from_file(file_path, format='mp3')
            if audio_segment.duration_seconds % 10 >2:
                total = int(np.ceil(audio_segment.duration_seconds / 10))
            else:
                total = int(np.floor(audio_segment.duration_seconds / 10))
            index_random = random.sample(range(0,total),total)
            for i in range(total):
                # 将音频10s切片，并以顺序进行命名

                if i == total-1:
                    audio_segment[i * 10000:].export(save_dir + file+'_'+ 'chunk{0}.wav'.format(index_random[i]),format="wav")
                else:
                    audio_segment[i * 10000:(i + 1) * 10000].export(save_dir + file+'_'+ 'chunk{0}.wav'.format(index_random[i]),format="wav")

save_dir_root = '/home/gaolili/deepLearning_project/Voiceprint_20201027/Input_file/whale/wav/'
audios_segments('/home/gaolili/deepLearning_project/Voiceprint_20201027/Input_file/whale/origin', save_dir_root)

'''
audio_segment = AudioSegment.from_file(file, format='wav')
total = int(audio_segment.duration_seconds / 10)
for i in range(total):
    # 将音频10s切片，并以顺序进行命名
    audio_segment[i*10000:(i+1)*10000].export("/home/gaolili/deepLearning_project/output/chunk{0}.wav".format(i), format="wav")
'''