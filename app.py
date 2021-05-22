from flask import Flask, render_template, request, jsonify

# # New test

import numpy as np
import librosa, random, torch, time, os
import pandas as pd
import torch.nn as nn
from torchvision import models
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
from torch.optim import Adam
import torch.nn.functional as F
import boto3
from pydub import AudioSegment


###############################################


app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():



    



    # return "<h1>Welcome to our server!</h1>"

#     return render_template("index.html", data="hey")

# @app.route("/prediction", methods=["POST"])
# def prediction():
    
    # audio = request.files['audio']
    # audio.save(os.path.join("./test_audio/", audio.filename)) //works for local file storage



    # LOCAL_NAME = request.headers.get('file-name')

    s3 = boto3.resource('s3')
    file_name = request.headers.get('file-name')
    BUCKET_NAME = 'deepbirdaudio'

    S3_FILE = file_name 
    LOCAL_NAME = file_name

    bucket = s3.Bucket(BUCKET_NAME)
    bucket.download_file(S3_FILE, LOCAL_NAME)

    # Remember to take pydub out of the requirement
    sound = AudioSegment.from_file(LOCAL_NAME)
    saved_name = LOCAL_NAME.rsplit( ".", 1 )[ 0 ]
    sound.export(saved_name + '.wav', format="wav")

    class Hparams():
        def __init__(self):
            #resnet50 resnext50_32x4d mobilenet_v2 efficientnet-b3  densenet121 densenet169 
            self.models_name = ['efficientnet-b0']
            self.chk = ['enet0_101_0.771_0.692.pt']
            self.count_bird = [265,265,265,265,150,265] #count birds|Количество птиц, 264 - all, 265 + nocall
            self.len_chack = [448,448,448,448,448,224] # The duration of the training files 448 = 5 second|Длительность обучающих файлов
            
            self.mel_folder = './mel/'
            self.n_fft = 892
            self.sr = 21952 
            self.hop_length=245
            self.n_mels =  224
            self.win_length = self.n_fft
            self.batch_size = 100 # 3 - b7, 8 - b5,  12 - b3, 25 - b0, 18 - b1 70
            self.lr = 0.001
            self.border = 0.5
            self.save_interval = 200 #Model saving interval
            # Список из count_bird птиц по пополуярности
            self.bird_count = pd.read_csv('./input/my-birdcall-datasets/bird_count.csv').ebird_code.to_numpy()        
            self.BIRD_CODE = {b:i for i,b in enumerate(self.bird_count)}
            self.INV_BIRD_CODE = {v: k for k, v in self.BIRD_CODE.items()}
            self.bird_count = self.bird_count[:self.count_bird[0]]


    hp = Hparams()
    def mono_to_color(X: np.ndarray,len_chack, mean=0.5, std=0.5, eps=1e-6):
        trans = transforms.Compose([transforms.ToPILImage(),
                                            transforms.Resize([hp.n_mels, len_chack]), transforms.ToTensor(),
                                            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        X = np.stack([X, X, X], axis=-1)
        V = (255 * X).astype(np.uint8)
        V = (trans(V)+1)/2
        return V
        
        
    def accuracy(y_true, y_pred):
        y_pred = torch.sigmoid(y_pred)
        y_pred = y_pred.detach().cpu().numpy()
        return f1_score(y_true > hp.border, y_pred > hp.border, average="samples")
        
        
    def get_melspectr(train_path):
        # Load file | Загружаем файл
        y, _ = librosa.load(train_path,sr=hp.sr,mono=True,res_type="kaiser_fast")

        # Create melspectrogram | Создать Мелспектрограмму
        spectr = librosa.feature.melspectrogram(y, sr=hp.sr, n_mels=hp.n_mels, n_fft=hp.n_fft, hop_length = hp.hop_length, win_length = hp.win_length, fmin = 300)
        return spectr.astype(np.float16)


    def random_power(images, power = 1.5, c= 0.7):
        images = images - images.min()
        images = images/(images.max()+0.0000001)
        images = images**(random.random()*power + c)
        return images
        

    class BirdcallNet( nn.Module):
        def __init__(self, name, num_classes=265):
            super(BirdcallNet, self).__init__()
            self.model = models.__getattribute__(name)(pretrained=False)
            if name in ["resnet50","resnext50_32x4d"]:
                self.model.fc = nn.Linear(2048, num_classes)
            elif name in ['resnet18','resnet34']:
                self.model.fc = nn.Linear(512, num_classes)
            elif  name =="densenet121":
                self.model.classifier = nn.Linear(1024, num_classes)
            elif name in ['alexnet','vgg16']:
                self.model.classifier[-1] = nn.Linear(4096, num_classes)
            elif name =="mobilenet_v2":
                self.model.classifier[1] = nn.Linear(1280, num_classes)
            #print(self.model)
        def forward(self, x):
            return self.model(x)

            
    def get_model(model_name,chk,count_bird):
        best_bird_count,best_score, epochs = 0,0,1
        all_loss, train_accuracy = [], []
        f1_scores,t_scores,b_scores = [],[],[]
        if not chk and model_name in ['efficientnet-b3','efficientnet-b0']:
            model = EfficientNet.from_pretrained(model_name, num_classes = count_bird)
            optimizer = Adam(model.parameters(), lr = hp.lr)
        else:
            models_names = ['alexnet','resnet50','resnet18','resnet34','mobilenet_v2','densenet121','resnext50_32x4d','densenet169']
            if model_name in models_names:
                model = BirdcallNet(model_name, hp.count_bird[0])
            elif model_name == 'mini':
                model = Classifier(hp.count_bird[0])
            else:
                model = EfficientNet.from_name(model_name, num_classes = count_bird)
                
            optimizer = Adam(model.parameters(), lr = hp.lr)
            # Load a checkpoint | Загрузить чекпоинт
            if chk:
                ckpt = torch.load('./input/my-birdcall-datasets/'+chk,map_location ='cpu')
                model.load_state_dict(ckpt['model'])
                epochs = int(ckpt['epoch']) + 1
                train_accuracy =  ckpt['train_accuracy'] 
                all_loss   = ckpt['all_loss'] 
                best_bird_count =  ckpt['best_bird_count'] 
                best_score   = ckpt['best_score']
                
                if 'optimizer' in ckpt:
                    optimizer.load_state_dict(ckpt['optimizer'])
                if 't_scores' in ckpt:
                    t_scores   = ckpt['t_scores']
                if 'f1_scores' in ckpt:
                    f1_scores   = ckpt['f1_scores']
                if 'b_scores' in ckpt:
                    b_scores   = ckpt['b_scores']
                print('Чекпоинт загружен: Эпоха %d Число обнаруженых птиц %d Score %.3f' % (epochs,best_bird_count,best_score))
        return model,optimizer, epochs, train_accuracy, all_loss, best_bird_count, best_score, t_scores, f1_scores, b_scores

    def generate(models, epochs, border,log_stat):
        start = time.time() 
        preds = []

        # Uploading a list of files for testing | Загружаем список файлов для тестирования
        # TEST_FOLDER = f'./test_audio/' // works for local file storage, have to change this line melspectr = get_melspectr
        
        audio_main = str(LOCAL_NAME)
        final_audio_id = audio_main.rsplit( ".", 1 )[ 0 ]

        test_info = pd.DataFrame(data = {
            'site': "site_1",
            'row_id': "site_1_" + final_audio_id + "_5",
            'seconds': [5],
            'audio_id': final_audio_id
        })

        # Looking for all unique audio recordings | Ищем все уникальные аудиозаписи
        unique_audio_id = test_info.audio_id.unique() 
        
        # Predict | Предсказываем
        for model in models:
            model.eval()
        with torch.no_grad():    
            for audio_id in unique_audio_id:
                # Getting a spectrogram | Получаем спектрограмму
                melspectr = get_melspectr(audio_id + '.wav')
                melspectr = librosa.power_to_db(melspectr, amin=1e-7, ref=np.max)
                melspectr = ((melspectr+80)/80).astype(np.float16)
                
                # Looking for all the excerpts for this sound | Ищем все отрывки для данного звука  
                test_df_for_audio_id = test_info.query(f"audio_id == '{audio_id}'").reset_index(drop=True)
                est_bird =np.zeros((265))
                probass = {}
                
                # Проходим по все отрывкам 
                for index, row in test_df_for_audio_id.iterrows():
                    # Getting the site, start time, and id | Получаем сайт, время начала и id
                    site = row['site']
                    start_time = row['seconds'] - 5
                    row_id = row['row_id']
                    mels = []
                    probas = None
                    
                    # Cut out the desired piece | Вырезаем нужный кусок
                    if site == 'site_1' or site == 'site_2':
                        start_index = int(hp.sr * start_time/hp.hop_length)
                        end_index = int(hp.sr * row['seconds']/hp.hop_length)                
                        y = melspectr[:,start_index:end_index]
                    else:
                        y = melspectr
                        
                    # cutting off the tail | отсекаю хвост
                    if (y.shape[1]%hp.len_chack[0]):
                        y = y[:,:-(y.shape[1]%448)]
                    
                    prob = []
                    for i,model in enumerate(models):
                        mels = []
                        probas = None                    
                        # Split into several chunks with the duration hp.len_chack | Разбиваем на несколько кусков длительностью hp.len_chack
                        ys = np.reshape(y, (hp.n_mels, -1, hp.len_chack[i]))
                        ys = np.moveaxis(ys, 1, 0)

                        # For each piece we make transformations | Для каждого куска делаем преобразования
                        for image in ys:
                            # Convert to 3 colors and normalize | Переводим в 3 цвета и нормализуем
                            image = image/image.max()
                            #image = image**0.85
                            #image = torch.from_numpy(np.stack([image, image, image])).float()
                            image = mono_to_color(image,hp.len_chack[i])
                            mels.append(image)

                        mels = np.stack(mels)                
                        
                        # Прохожу по всем batch
                        for n in range(0,len(mels),hp.batch_size):
                            if len(mels) == 1:
                                mel = np.array(mels)
                            else:
                                mel = mels[n:n+hp.batch_size]

                            mel = torch.from_numpy(mel)

                            # Predict | Получить выход модели
                            prediction = model(mel)
                            #prediction = F.softmax(prediction, dim=1)
                            prediction = torch.sigmoid(prediction)

                            # in numpy
                            proba = prediction.detach().cpu().numpy()

                            # Add zeros up to 265 | Добавить нули до 265
                            proba = np.concatenate((proba,np.zeros((proba.shape[0],265-proba.shape[1]))), axis=1)

                            # Adding to the array | Добавляю в массив
                            if not probas is None:
                                probas = np.append(probas, proba, axis = 0)
                            else:
                                probas = proba
                            if hp.len_chack[i] == 448:
                                probas = np.append(probas, proba, axis = 0)
                        prob.append(probas)

                    # Averaging the ensemble | Усредняю ансамбль
                    prob = np.stack(prob,axis=0)
                    prob = prob**2
                    proba = prob.mean(axis=0)#gmean(prob)/2 + prob.mean(axis=0)/2
                    proba = proba**(1/2)
                    
                    # If a bird is encountered in one segment, increase its probability in others
                    # Если встретилась птица в одном отрезке, увеличить её вероятность в других
                    for xx in proba:
                        z = xx.copy()
                        z[z<0.5] = 0
                        est_bird = est_bird + z/70
                        est_bird[(est_bird<0.15)&(est_bird>0)] = 0.15
        
                    # Dictionary with an array of all passages | Словарь с массивом всех отрывков
                    probass[row_id] = proba
                
                est_bird[est_bird>0.3] = 0.3
                for row_id,probas in probass.items():
                    prediction_dict = []
                    for proba in probas:
                        proba += est_bird
                        events = proba > border
                        labels = np.argwhere(events).reshape(-1).tolist()

                        # To convert in the name of the bird | Преобразовать в название птиц
                        if len(labels) == 0  or (264 in labels):
                            continue
                        else:
                            labels_str_list = list(map(lambda x: hp.INV_BIRD_CODE[x], labels))
                            for i in labels_str_list:
                                if i not in prediction_dict:
                                    print("Bird = " + i)
                                    prediction_dict.append(i)  
                        
                    # If birds are not predicted | Если не предсказываются птицы
                    if len(prediction_dict) == 0:
                        prediction_dict = "nocall"
                        print("nocall")
                    else:
                        prediction_dict = " ".join(prediction_dict)
            
                    # To add to the list | Добавить в список
                    preds.append([row_id, prediction_dict])

            # # Convert to DataFrame and save | Перевести в DataFrame и сохранить
            # preds = pd.DataFrame(preds, columns=['row_id', 'birds'])
            # preds.to_csv('submission.csv', index=False)
            
        print(preds)
        return preds


    # Main
    hp = Hparams()
    all_model = []
    for i in range(len(hp.models_name)):
        model,optimizer, epochs, train_accuracy, all_loss, best_bird_count, best_score, t_scores, f1_scores, b_scores = get_model(
                                                                                    hp.models_name[i],hp.chk[i],hp.count_bird[i])
        all_model.append(model)
    result = generate(all_model, epochs, hp.border, True)   

    return jsonify(result[0][1])

if __name__ == "__main__":
    # app.run(threaded=True, port=5000)
    app.run(debug=True)

