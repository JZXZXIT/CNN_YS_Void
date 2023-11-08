import pylab as p
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary
import numpy as np
import os
from datetime import datetime
import librosa
import librosa.display
import pywt
from Mymodel import AlexNet


class 声音相似度模型:
    def __init__(self, output_size: int, num_epochs: int, data_train: np.ndarray, data_label: np.ndarray, learning_rate: float = 0.001, batch_size: int = 32):
        '''

        :param output_size: 类别数量
        :param num_epochs: 训练轮次
        :param data_train: 训练数据
        :param data_label: 训练标签
        :param learning_rate: 学习率，别改
        :param batch_size: 批大小
        '''
        self.output_size = output_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.data_train = data_train
        self.data_label = data_label

    def 训练模型(self):
        ### 准备训练数据和标签
        input_size = self.data_train[0].shape
        print("输入shape：" + input_size)
        inputs = torch.Tensor(self.data_train)
        labels = torch.LongTensor(self.data_label)
        # 将模型和数据移动到相同的设备上
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AlexNet(num_classes=self.output_size).to(device)

        # 输出代码运行位置
        print(f"程序运行位置：{torch.cuda.get_device_name(device)}.")
        # 输出模型信息
        summary(model, input_size)
        print("\n")

        # 创建数据集和数据加载器
        train_dataset = TensorDataset(inputs, labels)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # 定义损失函数
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate)  # 定义优化器

        ### 模型训练
        训练信息 = []
        for epoch in range(self.num_epochs):
            total_loss = 0.0
            total_correct = 0

            for inputs_batch, labels_batch in train_dataloader:
                inputs_batch = inputs_batch.to(device)
                labels_batch = labels_batch.to(device)

                # 前向传播
                outputs = model(inputs_batch)
                loss = criterion(outputs, labels_batch)

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == labels_batch).sum().item()

            # 计算准确率和平均损失
            accuracy = total_correct / len(train_dataset)
            average_loss = total_loss / len(train_dataloader)

            # 打印训练信息
            输出 = f'第 [{epoch + 1}/{self.num_epochs}] 轮, 平均损失: {average_loss:.4f}, 准确率: {accuracy * 100:.2f}%'
            训练信息.append(输出)
            print(输出)

        ### 保存整个模型
        当前时间 = datetime.now()
        if not os.path.exists(f"modle_YSVoid_PT"):
            os.makedirs(f"modle_YSVoid_PT")
        torch.save(model, f'./modle_YSVoid_PT/{当前时间.strftime("%m_%d_%H_%M_%S")}_model.pth')
        with open(f"./modle_YSVoid_PT/{当前时间.strftime('%m_%d_%H_%M_%S')}_Introduce.txt", "w") as 介绍文件:
            介绍文件.write(f"文件名称：{当前时间.strftime('%m_%d_%H_%M_%S')}_model.pth\n文件保存时间：{当前时间}\n\n"
                       f"分类类别：{self.output_size}\n训练轮次：{self.num_epochs}\n初始学习率：{self.learning_rate}\n批大小：{self.batch_size}\ndata shape：{input_size}\n\n"
                       f"训练结果：\n")
            for i in range(len(训练信息)):
                介绍文件.write(f"{训练信息[i]}\n")

    @classmethod
    def 模型预测(cls, 文件路径):
        '''
        直接使用  声音相似度.模型预测(文件路径)  调用即可
        :param 文件路径: 输入需要预测的声音文件的路径，最好为.wav或.mp3文件（可被librosa库识别的文件）
        :return:
        '''
        def 声音降噪():
            '''
            使用小波阈值珐降噪，但是效果聊胜于无
            希望各位能够完善降噪的方法
            :return:
            '''
            y, sr = librosa.load(文件路径)
            # 定义小波函数
            wavelet = 'db4'
            # 计算功率谱密度
            S = np.abs(librosa.stft(y))
            psd = librosa.amplitude_to_db(S ** 2)
            # 删除白噪音
            mean_psd = np.mean(psd)
            std_psd = np.std(psd)
            if ((mean_psd - 3 * std_psd) > -100):
                # 小波阈值法降噪
                c = pywt.wavedec(y, wavelet, mode='symmetric')
                threshold = np.median(np.abs(c[-1])) / 0.6745
                for i in range(1, len(c)):
                    c[i] = pywt.threshold(c[i], threshold)
                y = pywt.waverec(c, wavelet)
            return y, sr

        ### 定义基本信息
        男角色 = ["五郎", "凯亚", "卡维", "托马", "提纳里", "枫原万叶", "流浪者", "温迪", "班尼特", "白术", "神里绫人", "米卡", "艾尔海森", "荒泷一斗", "行秋",
               "赛诺", "达达利亚", "迪卢克", "重云", "钟离", "阿贝多", "雷泽", "魈", "鹿野院平藏"]
        女角色 = ['七七', '丽莎', '久岐忍', '九条裟罗', '云堇', '优菈', '八重神子', '凝光', '刻晴', '北斗', '可莉', '坎蒂丝', '埃洛伊', '多莉',
               '夜兰', '妮露', '安柏', '宵宫', '早柚', '柯莱', '烟绯', '珊瑚宫心海', '珐露珊', '琴', '瑶瑶', '甘雨', '申鹤', '砂糖', '神里绫华', '纳西妲',
               '罗莎莉亚', '胡桃', '芭芭拉', '莫娜', '莱依拉', '菲谢尔', '诺艾尔', '辛焱', '迪奥娜', '迪希雅', '雷电将军', '香菱']
        hop_length = 512  # 步长
        n_fft = 2048  # FFT窗口大小
        n_mels = 128  # 梅尔滤波器数量
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ### 导入声音文件并进行预处理
        ## 导入与降噪
        data, sr = 声音降噪()
        ## 转换为梅尔频谱
        mel_spec = librosa.feature.melspectrogram(y=data, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        data = librosa.power_to_db(mel_spec, ref=np.max)  # 将功率谱转换为分贝
        ## 调整为网络的输入格式（1， 3， :， :）
        data = data[None, None, :, :]
        data = np.concatenate((data, data, data), axis=1)

        ### 输入网络进行预测
        ## 导入二分类模型，首先先区分声音的性别
        model = torch.load("./可用模型/男女二分类.pth")
        model = model.to(device)
        model.eval()
        ## 输入进二分类模型，根据模型需要（训练时使用的大小）调整输入大小
        data1 = torch.tensor(data[:, :, :, :800], dtype=torch.float32)
        data1 = data1.type(torch.FloatTensor).to(device)
        output = model(data1)
        ## 得到二分类模型的结果
        _, predicted = torch.max(output.data, 1)
        ## 根据声音的性别调用不同的模型进行预测
        最终概率 = {}
        if int(predicted.item()) == 0:
            model = torch.load("./可用模型/男角色24分类.pth")
            model = model.to(device)
            model.eval()
            data2 = torch.tensor(data[:, :, :, :800], dtype=torch.float32)
            data2 = data2.type(torch.FloatTensor).to(device)
            output = model(data2)
            # 计算每一类的概率
            probabilities = torch.softmax(output, dim=1)
            # 打印每一类的概率
            for i, prob in enumerate(probabilities[0]):
                概率 = prob.item() * 100  # 变为百分之多少
                概率 = round(概率, 1)  # 仅保留一位小数
                最终概率[男角色[i]] = 概率  # 保存概率
        else:
            model = torch.load("./可用模型/女角色42分类.pth")
            model = model.to(device)
            model.eval()
            data2 = torch.tensor(data[:, :, :, :800], dtype=torch.float32)
            data2 = data2.type(torch.FloatTensor).to(device)
            output = model(data2)
            # 计算每一类的概率
            probabilities = torch.softmax(output, dim=1)
            # 打印每一类的概率
            for i, prob in enumerate(probabilities[0]):
                概率 = prob.item() * 100
                概率 = round(概率, 1)
                最终概率[女角色[i]] = 概率
        ## 最多仅保留前三个概率最大的可能，且不输出概率为0的选项
        值 = list(最终概率.values())
        值.sort(reverse=True)
        输出 = ["", "", ""]
        for name, 概率 in 最终概率.items():
            if 概率 == 0:
                continue
            if 概率 == 值[0]:
                输出[0] = f"{name}：{概率}%"
            elif 概率 == 值[1]:
                输出[1] = f"{name}：{概率}%"
            elif 概率 == 值[2]:
                输出[2] = f"{name}：{概率}%"
        for i in range(3):
            if 输出[i] == "":
                continue
            print(输出[i])

class 制作数据集:
    @classmethod
    def 特征提取_梅尔频谱(cls, 文件路径="./原神语音包/"):
        import librosa.display

        ### 导入基本信息
        文件ss = os.listdir(文件路径)
        文件s = [[], []]  # 男，女
        os.makedirs("./数据集/梅尔频谱_按角色分_PC/", exist_ok=True)  # 若不存在该文件夹，则创建文件夹
        ## 区分男女
        for 文件 in 文件ss:
            if 文件.split("_")[0] == "男":
                文件s[0].append(f"{文件路径}{文件}")
            else:
                文件s[1].append(f"{文件路径}{文件}")
        性别 = ["男", "女"]

        ### 处理
        for i in range(len(文件s)):
            for 文件 in 文件s[i]:
                文件名称 = 文件[8:-4]  # 获取角色名称
                data1 = np.zeros((0, 3, 128, 800))  # 创建一个空的np数组，用以存储处理后的数据
                print(f"正在处理：{文件名称}")
                data, sr = librosa.load(文件, sr=None)  # 导入音频文件
                ## 划窗与分帧
                窗口大小 = sr * 10
                步长 = sr * 5
                结束点 = 窗口大小
                音频长度 = len(data)
                while 结束点 < 音频长度:
                    ## 计算梅尔频谱
                    data_r = librosa.feature.melspectrogram(y=data[结束点-窗口大小:结束点], sr=sr, n_fft=2048, hop_length=512,
                                                              n_mels=128)
                    data_r = librosa.power_to_db(data_r, ref=np.max)  # 将功率谱转换为分贝
                    ## 调整为网络可接受的格式
                    data_r = data_r[None, None, :, :800]
                    data_r = np.concatenate((data_r, data_r, data_r), axis=1)  # 特征维叠加到3
                    data1 = np.concatenate((data1, data_r), axis=0)
                    ## 更新结束点
                    结束点 += 步长
                np.save(f"./数据集/梅尔频谱_按角色分_PC/{性别[i]}/{性别[i]}_{文件名称}.npy", data1)
                with open(f"./数据集/梅尔频谱_按角色分_PC/{性别[i]}/Introduce.txt", "a") as 记录文件:
                    记录文件.write(f"文件名称：{性别[i]}_{文件名称}.npy\n对应角色：{文件名称}\nshpe：{data1.shape}\n-------------------------------------------\n\n\n")

    @classmethod
    def 整合数据集_男女二分类(cls):
        '''
        仅示例了整理为男女二分类的数据集的过程，可以仿照这个函数写整理为多分类数据集的函数
        :return:
        '''
        ### 获取基本信息
        父路径 = r"./数据集/梅尔频谱_按角色分_PC/"
        性别 = ["男/", "女/"]

        ### 导入数据
        data = [np.zeros((0, 3, 128, 800)), np.zeros((0, 3, 128, 800))]  # 男角色数据，女角色数据
        for i in range(2):
            文件s = os.listdir(父路径 + 性别[i])
            for 文件 in 文件s:
                print(f"正在处理：{文件}")
                LSdata = np.load(父路径 + 性别[i] + 文件)
                data[i] = np.concatenate((data[i], LSdata), axis=0)

        ### 打上标签
        ## 初始标签
        label = [
            np.zeros((data[0].shape[0], )),  # 0代表男
            np.ones((data[1].shape[0], ))  # 1代表女
        ]
        ## 整理数据
        data = np.concatenate((data[0], data[1]), axis=0)
        label = p.concatenate((label[0], label[1]), axis=0)
        ## 用随机索引打乱数据集和标签（可以选择不打乱，因为在网络中会打乱的）
        random_state = np.random.RandomState(seed=42)
        shuffle_indices = random_state.permutation(data.shape[0])
        data = data[shuffle_indices, :, :, :]
        label = label[shuffle_indices]

        ### 保存
        np.savez("./数据集/数据And标签.npz", data=data, label=label)

        print(data.shape)