import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary
import pickle
import numpy as np
import os
from datetime import datetime
import scipy.io as sio
import math
import scipy.signal as signal
import librosa
import librosa.display
import pywt
import soundfile as sf
from pydub import AudioSegment
from matplotlib.font_manager import FontProperties

### 这个文件没有整理，里边保存了很多功能，可以尝试调用

def 绘制梅尔频谱():
    filename = "./原神语音包/男_魈.mp3"
    data, sr = librosa.load(filename, sr=None)

    hop_length = 512  # 步长
    n_fft = 2048  # FFT窗口大小
    n_mels = 128  # 梅尔滤波器数量

    # 计算梅尔频谱
    mel_spec = librosa.feature.melspectrogram(y=data, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)  # 将功率谱转换为分贝
    print(mel_spec_db.shape)

    # 可以使用librosa.display库将梅尔频谱可视化
    librosa.display.specshow(mel_spec_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.show()

def 使用librosa库():
    filename = "./原神语音包/男_魈.mp3"

    # 获取采样率
    sample_rate = librosa.get_samplerate(filename)
    print("采样率:", sample_rate)

    # 获取持续时间
    duration = librosa.get_duration(path=filename)
    print("持续时间:", duration)

    # 获取数据类型
    data, sr = librosa.load(filename, sr=None)
    data_type = data.dtype
    print("数据类型:", data_type)

def 模型预测():
    # filename = "./原神语音包/男_魈.mp3" # sr=44100
    # data, sr = librosa.load(filename, sr=None)
    类型 = ["男声","女声"]
    男角色 = ["五郎","凯亚","卡维","托马","提纳里","枫原万叶","流浪者","温迪","班尼特","白术","神里绫人","米卡","艾尔海森","荒泷一斗","行秋","赛诺","达达利亚","迪卢克","重云","钟离","阿贝多","雷泽","魈","鹿野院平藏"]
    女角色 = []
    data , sr = 声音降噪() # sr=22050
    # data = data[5 * sr:6 * sr]
    hop_length = 512  # 步长
    n_fft = 2048  # FFT窗口大小
    n_mels = 128  # 梅尔滤波器数量
    # 计算梅尔频谱
    mel_spec = librosa.feature.melspectrogram(y=data, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    data = librosa.power_to_db(mel_spec, ref=np.max)  # 将功率谱转换为分贝
    data = data[None, None, :, :]
    data = np.concatenate((data, data, data), axis=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load("./可用模型/男角色24分类.pth")
    model = model.to(device)
    model.eval()

    data = torch.tensor(data, dtype=torch.float32)
    data = data.type(torch.FloatTensor).to(device)
    output = model(data)

    # _, predicted = torch.max(output.data, 1)
    # print(f"预测结果为：{男角色[predicted.item()]}")
    # 计算每一类的概率
    probabilities = torch.softmax(output, dim=1)
    # 打印每一类的概率
    for i, prob in enumerate(probabilities[0]):
        概率 = prob.item() * 100
        概率 = round(概率,1)
        print(f"{男角色[i]}：{概率}%")

def 声音降噪():
    '''
    没啥用
    :return:
    '''
    y, sr = librosa.load('./combined_audio.wav')

    # Define wavelet function
    wavelet = 'db4'

    # Calculate power spectral density
    S = np.abs(librosa.stft(y))
    psd = librosa.amplitude_to_db(S ** 2)

    # Detect white noise
    mean_psd = np.mean(psd)
    std_psd = np.std(psd)
    if ((mean_psd - 3 * std_psd) > -100):
        print('检测到白噪音，开始滤除……')
        # Noise reduction using wavelet thresholding
        c = pywt.wavedec(y, wavelet, mode='symmetric')
        threshold = np.median(np.abs(c[-1])) / 0.6745
        for i in range(1, len(c)):
            c[i] = pywt.threshold(c[i], threshold)
        y = pywt.waverec(c, wavelet)

        # Save denoised audio data
        sf.write('./处理后.wav', y, sr)
        print("滤除成功！")
    return y, sr

def 将acc变为wav():
    # 加载 AAC 文件
    aac_file = "./人声语音/1.m4a"
    audio = AudioSegment.from_file(aac_file, format="m4a")

    # 将音频保存为临时的 WAV 文件
    temp_wav_file = "./人声语音/1.wav"
    audio.export(temp_wav_file, format="wav")

def 噪音叠加():
    # 加载人声和噪声音频文件
    voice_audio = AudioSegment.from_file("./人声语音/0.mp3")
    noise_audio = AudioSegment.from_file("./噪音/住宅区灯光氛围循环-噪音_爱给网_aigei_com.mp3")
    noise_audio += 10 # 将噪声音量提高

    # 将两个音频文件叠加在一起
    combined_audio = voice_audio.overlay(noise_audio)

    # 输出叠加后的音频文件
    combined_audio.export("噪音叠加后.wav", format="wav")

def 音频信号分帧处理(音频地址):
    '''
    对音频信号进行分帧，在分帧过程中添加为汉明窗，用以加权
    :param audio_signal:输入信号
    :param frame_size:第一次加窗的窗口长度
    :param hop_size:滑动距离
    :return:处理后的二维矩阵(n,x,100)
    '''
    audio_signal, sr = librosa.load(音频地址)
    frame_size = sr  # 帧大小
    hop_size = (sr // 3) * 2  # 帧之间的跳跃大小
    x = ((frame_size - 100) // 75) + 1
    frames = np.zeros((1,x,100))
    num_samples = len(audio_signal)
    window = np.hanning(100)  # 使用汉宁窗函数
    start = 0
    while start + frame_size <= num_samples:
        frame = audio_signal[start:start+frame_size]
        start1 = 0
        窗口长度 = len(frame)
        datar = np.zeros((1,100))
        while start1 + 100 <= 窗口长度:
            data = frame[start1:start1+100]
            data *= window  # 对帧数据应用汉宁窗
            datar = np.concatenate((datar,data[None,:]),axis=0)
            start1 += 75
        frame = np.delete(datar,0,axis=0)
        frames = np.concatenate((frames,frame[None,:]),axis=0)
        start += hop_size
    frames = np.delete(frames,0,axis=0)
    return frames

def 时频变频谱变时频():
    '''
    最终的信号确实有区别，但是听不出来
    :return:
    '''
    # 设置中文字体
    font = FontProperties(fname='simsun.ttc')
    plt.rcParams['font.family'] = font.get_name()  # 设置字体系列
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    y, sr = librosa.load('./人声语音/1.wav')
    原始信号 = y[:100000]
    # sf.write('./1.wav', 原始信号, sr)

    频域信号 = np.fft.fft(原始信号)
    # print(原始信号)

    还原信号 = np.real(np.fft.irfft(频域信号, n=100000))
    # print(还原信号)

    if np.any(原始信号 == 还原信号):
        print("hhh")

    # sf.write('./3.wav', 还原信号, sr)

    plt.figure(figsize=(12, 18))

    plt.subplot(2, 2, 1)
    plt.plot(range(len(原始信号)), 原始信号)
    plt.title("原始信号")

    plt.subplot(2, 2, 2)
    plt.plot(range(len(频域信号)), 频域信号)
    plt.title("频域信号")

    plt.subplot(2, 2, 3)
    plt.plot(range(len(还原信号)), 还原信号)
    plt.title("还原信号")

    plt.tight_layout()
    plt.show()

def 原数据分帧():
    def 分帧(y):
        x = 0
        frames = np.zeros((1, 1000))
        while x + 1000 <= 1000000:
            frame = y[x:x + 1000]
            frame = frame[None, :]
            frames = np.concatenate((frames, frame), axis=0)
            x += 1000
        frames = np.delete(frames, 0, axis=0)
        return frames

    data, sr = librosa.load('./加噪音频/0_女_琴.wav')
    target, _ = librosa.load('./原神语音包/女_琴.mp3')

    data = data[:10000000]
    target = target[:10000000]

    print(data.shape)
    print(target.shape)

    data = 分帧(data)
    target = 分帧(target)

    print(data.shape)
    print(target.shape)

    np.save("./数据集/女_琴原数据分帧/data.npy", data)
    np.save("./数据集/女_琴原数据分帧/target.npy", target)

def 将单个数据集整合为一个大数据集():
    文件 = os.listdir("./数据集/分帧FFT/原声")
    文件s = []
    for i in 文件:
        if i[-4:] == ".npy":
            文件s.append(i)
    print(文件s)
    data_target = np.zeros((0, 1, 1, 10000))
    for 文件 in 文件s:
        data = np.load(f"./数据集/分帧FFT/原声/{文件}")
        data_target = np.concatenate((data_target, data), axis=0)
    print(data_target.shape)
    np.save("./数据集/分帧FFT/整合/原声.npy", data_target)
    data_target = None

    data_train0 = np.zeros((0, 1, 1, 10000))
    for 文件 in 文件s:
        data = np.load(f"./数据集/分帧FFT/噪声0/{文件}")
        data_train0 = np.concatenate((data_train0, data), axis=0)
    print(data_train0.shape)
    np.save("./数据集/分帧FFT/整合/噪声0.npy", data_train0)
    data_train0 = None

    data_train1 = np.zeros((0, 1, 1, 10000))
    for 文件 in 文件s:
        data = np.load(f"./数据集/分帧FFT/噪声1/{文件}")
        data_train1 = np.concatenate((data_train1, data), axis=0)
    print(data_train1.shape)
    np.save("./数据集/分帧FFT/整合/噪声1.npy", data_train1)

if __name__ == "__main__":
    # 模型预测()
    # 声音降噪()
    将acc变为wav()
    # 绘制梅尔频谱()
    pass