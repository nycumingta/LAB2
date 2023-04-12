#%matplotlib inline
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data

import matplotlib.pyplot as plt
import pickle
from scipy.fftpack import fft,ifft
from scipy.signal import butter, lfilter, freqz

device = torch.device("cuda:0")
batch_size = 512  # 512
EPOCH = 3000
LR = 0.001
Lsize = 736
WeightDecay = 0.02
DropoutP = 0.25
cutoff = 10
#FFTAdd  FIRAdd  None
DataType = 'None'


def read_bci_data():
    S4b_train = np.load('S4b_train.npz')
    X11b_train = np.load('X11b_train.npz')
    S4b_test = np.load('S4b_test.npz')
    X11b_test = np.load('X11b_test.npz')

    train_data = np.concatenate((S4b_train['signal'], X11b_train['signal']), axis=0)
    train_label = np.concatenate((S4b_train['label'], X11b_train['label']), axis=0)
    test_data = np.concatenate((S4b_test['signal'], X11b_test['signal']), axis=0)
    test_label = np.concatenate((S4b_test['label'], X11b_test['label']), axis=0)

    train_label = train_label - 1    
    test_label = test_label -1    
    train_data = np.transpose(np.expand_dims(train_data, axis=1), (0, 1, 3, 2))
    test_data = np.transpose(np.expand_dims(test_data, axis=1), (0, 1, 3, 2))    

    mask = np.where(np.isnan(train_data))
    train_data[mask] = np.nanmean(train_data)

    mask = np.where(np.isnan(test_data))
    test_data[mask] = np.nanmean(test_data)    

    print(train_data.shape, train_label.shape, test_data.shape, test_label.shape)
    return train_data, train_label, test_data, test_label

def FFT_trans(x,y,style=''):
    yy=fft(y) #快速傅立葉轉換
    yreal = yy.real #實數部分
    yimag = yy.imag #虛數部分
    
    yf=abs(fft(y)) #abs
    yf1=abs(fft(y))/len(x) #normalization
    yf2 = yf1[range(int(len(x)/2))]  #take 1/2
    
    xf = np.arange(len(y)) #frequence
    xf1 = xf
    xf2 = xf[range(int(len(x)/2))] #take 1/2
    
    if (style == 'TwoSides'):
        return xf,yf
    elif (style == 'TwoSidesNor'):
        return xf1,yf1
    else:
        return xf2,yf2 

def DataAddFFTInfo(X):
    samplerate = 125 #The EEG was sampled with 125 Hz
    cycle = 1/samplerate
    arrayTmplay =[]
    for index in range(len(X)):
        arrayTmp = []
        ch1 = X[index][0][0]
        ch2 = X[index][0][1]
        EEG_time = np.arange(0, cycle*len(ch1), cycle)
        x,ch1fft = FFT_trans(EEG_time,ch1,'TwoSides')
        x,ch2fft = FFT_trans(EEG_time,ch2,'TwoSidesNor')
        arrayTmp = np.array([[np.vstack((ch1,ch2,ch1fft,ch2fft))]])

        if (index == 0):
            arrayTmplay = arrayTmp
        else :
            arrayTmplay = np.vstack((arrayTmplay,arrayTmp))
    return arrayTmplay  

def DataFilter(X,cutoff):
    order = 6
    fs = 125.0       # sample rate, Hz
    #cutoff = 10  # desired cutoff frequency of the filter, Hz  EEG 0.1~70 Hz
    ch_fil = butter_lowpass_filter(X, cutoff, fs, order)
    return ch_fil  

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


class EEGNet_ELU(nn.Module):
    def __init__(self):
        
        super(EEGNet_ELU, self).__init__()
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, (1, 51), padding = (0,25), bias=True),
            nn.BatchNorm2d(16, eps=0.00001, momentum=0.1 , affine=True, track_running_stats=True)
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, (2, 1), groups =16,bias=True ),
            nn.BatchNorm2d(32, 0.00001, affine=True),
            nn.ELU(alpha=1.0),
            nn.AvgPool2d((1,4)),
            nn.Dropout(p=DropoutP)
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, (1, 15), padding = (0,7), bias=True ),
            nn.BatchNorm2d(32, 0.00001, affine=True),
            nn.ELU(alpha=1.0),
            nn.AvgPool2d((1,8)),
            nn.Dropout(p=DropoutP)
        )
        self.classify = nn.Sequential(
            nn.Linear(Lsize,2, bias=True)
        )
        
    def forward(self, x):
        #Layer firstconv
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = x.view(x.size(0),-1)
        return self.classify(x)

class EEGNet_ReLU(nn.Module):
    def __init__(self):
        
        super(EEGNet_ReLU, self).__init__()
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, (1, 51), padding = (0,25), bias=True),
            nn.BatchNorm2d(16, eps=0.00001, momentum=0.1 , affine=True, track_running_stats=True)
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, (2, 1), groups =16,bias=True ),
            nn.BatchNorm2d(32, 0.00001, affine=True),
            nn.ReLU(),
            nn.AvgPool2d((1,4)),
            nn.Dropout(p=DropoutP)
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, (1, 15), padding = (0,7), bias=True ),
            nn.BatchNorm2d(32, 0.00001, affine=True),
            nn.ReLU(),
            nn.AvgPool2d((1,8)),
            nn.Dropout(p=DropoutP)
        )
        self.classify = nn.Sequential(
            nn.Linear(Lsize,2, bias=True)
        )
        
    def forward(self, x):
        #Layer firstconv
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = x.view(x.size(0),-1)
        return self.classify(x)
    

class EEGNet_LeakyReLU(nn.Module):
    def __init__(self):
        
        super(EEGNet_LeakyReLU, self).__init__()
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, (1, 51), padding = (0,25), bias=True),
            nn.BatchNorm2d(16, eps=0.00001, momentum=0.1 , affine=True, track_running_stats=True)
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, (2, 1), groups =16,bias=True ),
            nn.BatchNorm2d(32, 0.00001, affine=True),
            nn.LeakyReLU(),
            nn.AvgPool2d((1,4)),
            nn.Dropout(p=DropoutP)
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, (1, 15), padding = (0,7), bias=True ),
            nn.BatchNorm2d(32, 0.00001, affine=True),
            nn.LeakyReLU(),
            nn.AvgPool2d((1,8)),
            nn.Dropout(p=DropoutP)
        )
        self.classify = nn.Sequential(
            nn.Linear(Lsize,2, bias=True)
        )
        
    def forward(self, x):
        #Layer firstconv
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = x.view(x.size(0),-1)
        return self.classify(x)    
    

X_train, y_train, X_test, y_test = read_bci_data()    
if (DataType == 'FFTAdd'):
    X_train,X_test = DataAddFFTInfo(X_train), DataAddFFTInfo(X_test)
if (DataType == 'FIRAdd'):
    X_train,X_test = DataFilter(X_train,cutoff), DataFilter(X_test,cutoff)

def accuracy(y_true, y_pred):
    y_pred[y_pred >=0.5] = 1
    y_pred[y_pred < 0.5] = 0
    return (y_pred == y_true).mean()

def evaluate(model, test_x, test_y, params = ["acc"], t="train"):
    loss_fn = torch.nn.CrossEntropyLoss()

    results = []
    batch_size = 100    
    predicted = []
    model.eval()
    if t == "test":
        test_x = torch.from_numpy(test_x).to(device, dtype=torch.float)
        test_y = torch.from_numpy(test_y).to(device, dtype=torch.long)
    test_output = model(test_x).to(device)
    loss = loss_fn(test_output, test_y)
    pred_y = torch.max(test_output, 1)[1].data.cpu().numpy().squeeze()
    
#     for i in range(int(len(X)/batch_size)):
#         s = i*batch_size
#         e = i*batch_size+batch_size
        
#         inputs = Variable(torch.FloatTensor(X[s:e]).cuda())
#         #Y = Variable(torch.FloatTensor(Y[s:e]).cuda())      
        
#         pred = model(inputs)        
#         predicted.append(pred.data.cpu().numpy())
        
        
#     inputs = Variable(torch.FloatTensor(X).cuda())
    
#     predicted = model(inputs)
    
#     predicted = torch.max(predicted, 1)[1].data.cpu().numpy()
    
    for param in params:
        if param == 'acc':
            results.append(accuracy(pred_y, test_y.cpu().numpy()))
#             results.append(accuracy_score(Y, np.round(predicted)))
#         if param == "auc":
#             results.append(roc_auc_score(Y, predicted))
#         if param == "recall":
#             results.append(recall_score(Y, np.round(predicted)))
#         if param == "precision":
#             results.append(precision_score(Y, np.round(predicted)))
#         if param == "fmeasure":
#             precision = precision_score(Y, np.round(predicted))
#             recall = recall_score(Y, np.round(predicted))
#             results.append(2*precision*recall/ (precision+recall))
    return results
    

X_train = torch.from_numpy(X_train).to(device, dtype=torch.float)
y_train = torch.from_numpy(y_train).to(device, dtype=torch.long)
torch_dataset = Data.TensorDataset(X_train, y_train)

train_loader = Data.DataLoader(dataset=torch_dataset, batch_size=batch_size)

def gotrain(net,batch_size,EPOCH,LR,model):
    print('-----------------------')
    print(net)
    print('-----------------------')
    optimizer = torch.optim.Adam(net.parameters(), weight_decay=WeightDecay, lr=LR)
    # loss_fn = torch.nn.MSELoss(reduction='none') ## MSE
    loss_fn = torch.nn.CrossEntropyLoss()
    pltTrACC = []
    pltTsACC = []
    pltLOSS = []
    global X_train, y_train, X_test, y_test
    testMaxacc = 0.8
    for epoch in range(EPOCH):
        running_loss = 0.0
        net.train() ##
#         for i in range(int(len(X_train)/batch_size-1)):
#             s = i*batch_size
#             e = i*batch_size+batch_size

#             X = Variable(torch.FloatTensor(X_train[s:e]).cuda())
#             Y = Variable(torch.FloatTensor(y_train[s:e]).cuda())
        for step, (batch_x, batch_y) in enumerate(train_loader):
            X = batch_x
            labels = batch_y
            
            outputs = net(X).to(device)

#             outputs = outputs.reshape(-1)

#             labels = Y.reshape(-1)

            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss
        params = ["acc", "auc", "fmeasure"]
        ## totalloss = running_loss.data.cpu().numpy().sum()
        totalloss = running_loss

        testacc = evaluate(net, X_test, y_test, params, t="test")
        trainacc = evaluate(net, X_train, y_train, params, t="train")
#         totalloss = round(totalloss[0],4)
        testacc = round(testacc[0],4)
        if(testacc > testMaxacc):
            testMaxacc = testacc
            torch.save(net,model+'_'+str(EPOCH)+'_'+str(LR)+'_'+str(testacc)+'.pkl')
        trainacc = round(trainacc[0],4)
        pltLOSS.append(totalloss)
        pltTsACC.append(testacc)
        pltTrACC.append(trainacc)
        if epoch % 50 == 0:
            print(model+" epoch:"+str(epoch)+' Training Loss:'+str(totalloss)+' Train/Test ACC: '+str(trainacc)+'/'+str(testacc))
    torch.save(net,model+'_'+str(EPOCH)+'_'+str(LR)+'_'+str(testacc)+'.pkl')
    return pltLOSS,pltTrACC,pltTsACC
    
def pltLossAcc(loss,trainacc,testacc,model):
    plt.figure(figsize=(40,20))
    plt.subplot(2,2,1)
    plt.xlabel('time(epoch)')
    plt.ylabel('LOSS')
    plt.plot(loss)

    plt.subplot(2,2,2)
    plt.xlabel('time(epoch)')
    plt.ylabel('ACC')
    plt.plot(testacc,c='r',label='testacc')
    plt.plot(trainacc,c='b',label='trainacc')
    plt.legend()
    plt.savefig(model+".png")
    plt.show()


loss_fn = torch.nn.CrossEntropyLoss()

for i in range(1, 2):
    print('LR='+str(LR))
    Ttrain = []
    Ttest = []
    netELU = EEGNet_ELU().to(device)
    MODEL = 'EEGNet_ELU'
    loss,trainacc,testacc = gotrain(netELU,batch_size,EPOCH,LR,MODEL)
    pltLossAcc(loss,trainacc,testacc,MODEL)
    Ttest.append(testacc)
    Ttrain.append(trainacc)

    netReLU = EEGNet_ReLU().to(device)
    MODEL = 'EEGNet_ReLU'
    loss,trainacc,testacc = gotrain(netReLU,batch_size,EPOCH,LR,MODEL)
    pltLossAcc(loss,trainacc,testacc,MODEL)
    Ttest.append(testacc)
    Ttrain.append(trainacc)

    netLeakyReLU = EEGNet_LeakyReLU().to(device)
    MODEL = 'EEGNet_LeakyReLU'
    loss,trainacc,testacc = gotrain(netLeakyReLU,batch_size,EPOCH,LR,MODEL)
    pltLossAcc(loss,trainacc,testacc,MODEL)
    Ttest.append(testacc)
    Ttrain.append(trainacc)

    plt.figure(figsize=(40,20))
    plt.plot(Ttest[0],c='r',label='ELU Test')
    plt.plot(Ttrain[0],c='g',label='ELU Train')
    plt.plot(Ttest[1],c='b',label='ReLU Test')
    plt.plot(Ttrain[1],c='y',label='ReLU Train')
    plt.plot(Ttest[2],c='k',label='LeakyReLU Test')
    plt.plot(Ttrain[2],c='c',label='LeakyReLU Train')
    plt.legend()
    pacc = round(Ttest[0][0],4)
    plt.savefig('EEGNetNet'+'_'+str(EPOCH)+'_'+str(LR)+'_'+str(pacc)+".png")
    plt.show()
    # LR += 0.001
