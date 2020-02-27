import numpy as np
import random
import math
import time
import struct
import matplotlib.pylab as plt
from PHPFina import PHPFina,humanDate
import tensorflow as tf
import sys
import copy
from pandas import DataFrame
from sklearn import linear_model

def InitializeFeed(nb,step,start):
    feed=PHPFina(nb,step)
    feed.getMetas()
    feed.setStart(start)
    return feed

# given some PHPFina feeds, a period and a start (as a unix timestamp) both in seconds
def GoToTensor(params,step,start,nbsteps):
    print("going to tensor for {} feeds".format(len(params)))
    float_data=np.zeros((nbsteps,len(params)))
    for i in range(len(params)):
        print("feed number {}".format(i))
        feed=InitializeFeed(params[i]["id"],step,start)
        if params[i]["action"]=="smp":
            feed.getDatas(nbsteps)
        elif params[i]["action"]=="acc":
            feed.getKwh(nbsteps)
        if len(feed._datas):
            float_data[:,i]=feed._datas[0:nbsteps]
        else:
            return False
    return float_data

def plot_train_history(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()
    plt.show()

class BuildingZone():
    # history_size and target_size are integers
    # we want history_size steps of history
    # we want target_size steps in a single prediction
    # if target_size is 1, prediction is the next point
    # if target_size > 1, there is target_size points in the prediction
    def __init__(self,step,history_size,target_size):
        self._step=step
        self._history_size=history_size
        self._target_size=target_size
        self._mean=[]
        self._std=[]
        self._train_datas=[]
        self._train_labels=[]
        self._val_datas=[]
        self._val_labels=[]
        self._debug=False
        self._lab = ["out. temp","indoor temp", "Kwh"]
        self._col = ["blue", "green", "red"]
        self._MLAintercept=0
        self._MLAcoef=[]
        self._MLAregularize=True
        self._LSTMmodel = tf.keras.models.Sequential()

    def MLAprepare(self, datas, labelsToPhysicsValue=False):
        """
        prepare datas for multilinear Regression

        :param datas: tensor created by GoToTensor on the basis of some PHPFina timeseries
        :return: 2 numpy arrays

        Only the results of the ml regression are stored in the class
        you have to prepare AND to transmit the result of preparation to the predict method
        MLA_datas, MLA_labels = MLAprepare(datas)
        MLApredict(MLA_datas,nbset,goto)
        """
        clone=copy.deepcopy(datas)
        if self._MLAregularize:
            clone=(clone-self._mean)/self._std
        msize=clone.shape[0]-self._history_size
        MLA_datas=np.zeros((msize,clone.shape[1]*self._history_size))
        indice=0
        for j in range(clone.shape[1]):
            for i in range(self._history_size):
                MLA_datas[:,indice]=clone[i:i+msize,j]
                indice+=1
        MLA_labels=clone[self._history_size:clone.shape[0],1]
        if self._MLAregularize and labelsToPhysicsValue:
            MLA_labels = np.array(MLA_labels)*self._std[1]+self._mean[1]
        return MLA_datas, MLA_labels

    def MLAfit(self, datas, regularize=True):
        """
        multilinear regression
        :param datas: tensor created by GoToTensor on the basis of some PHPFina timeseries
        :param regularize: boolean - if true datas are regularized with the mean/std technique
        """
        self._MLAregularize=regularize
        MLA_datas, MLA_labels=self.MLAprepare(datas)
        regr = linear_model.LinearRegression()
        regr.fit(MLA_datas, MLA_labels)
        self._MLAintercept=regr.intercept_
        self._MLAcoef=regr.coef_
        if self._debug:
            print("MLA datas size {}".format(MLA_datas.shape))
            print("MLA labels size {}".format(len(MLA_labels)))
            print('Intercept: \n', self._MLAintercept)
            print('Coefficients: \n', self._MLAcoef)

    def MLApredict(self,datas,nbset,goto):
        """
        executes prediction(s) step by step with the multilinear method
        prediction 10 is made using all 9 previous predictions

        :param datas: an array with the samples to use
        :param nbset: the first sample to use for prediction
        :param goto: the number of prediction(s) to realize
        if goto is set to 1, the method will realize only one prediction
        """
        # pred is an array to store the predictions step by step
        pred=[]
        for k in range(goto):
            # in each sample, we find the indoor T values from index history_size to 2*history_size-1
            # the number of indoor T values is history_size
            # l is the size of the pred array
            # to make the predictions step by step :
            # - if l < history_size, we have to replace the last l T values by the predicted ones
            # - if l >= history_size, we have to replace the whole history_size values in the sample by the predicted ones
            sample=copy.deepcopy(datas[nbset+k])
            if self._debug:
                print(pred)
                print("before injection")
                print(sample)
            if len(pred)>0:
                if len(pred)>self._history_size:
                    simTs=pred[-self._history_size:]
                else:
                    simTs=pred
                sample[2*self._history_size-min(len(pred),self._history_size):2*self._history_size]=simTs
            if self._debug:
                print("after injection")
                print(sample)
                input("press any key")
            prediction=np.sum(self._MLAcoef*sample)+self._MLAintercept
            pred.append(prediction)
        if self._MLAregularize:
            pred=np.array(pred)*self._std[1]+self._mean[1]
        return pred

    def MLAviewWeights(self):
        plt.subplot(111)
        plt.title("Multilinear regression - coefficient {}".format(self._MLAintercept))
        plt.plot(self._MLAcoef)
        plt.show()

    def CalcMeanStd(self, datas):
        self._mean = datas.mean(axis=0)
        self._std = datas.std(axis=0)
        print(self._mean)
        print(self._std)

    def ClearSets(self, train=True):
        """
        clear datas and labels arrays
        """
        if train:
            self._train_datas=[]
            self._train_labels=[]
        else:
            self._val_datas=[]
            self._val_labels=[]

    def AddSets(self, datas, regularize=True, forTrain=True):
        clone=copy.deepcopy(datas)
        if regularize:
            clone=(clone-self._mean)/self._std
        # l is the number of datasets we can construct with datas
        l=clone.shape[0]-self._target_size-self._history_size
        rows=np.arange(self._history_size,self._history_size+l,1)
        if forTrain:
            np.random.shuffle(rows)
        if self._debug:
            print(rows.shape)
            print(rows)
            input("press any key")
        for j in range(len(rows)):
            indices = range(rows[j]-self._history_size, rows[j])
            if forTrain:
                self._train_datas.append(clone[indices])
                self._train_labels.append(clone[rows[j]:rows[j]+self._target_size,1])
            else:
                self._val_datas.append(clone[indices])
                self._val_labels.append(clone[rows[j]:rows[j]+self._target_size,1])

    def LSTMfit(self, name):
        tdat=np.array(self._train_datas)
        tlab=np.array(self._train_labels)
        vdat=np.array(self._val_datas)
        vlab=np.array(self._val_labels)
        self._LSTMmodel.add(tf.keras.layers.LSTM(32, return_sequences=True, input_shape=tdat.shape[-2:]))
        self._LSTMmodel.add(tf.keras.layers.LSTM(16, activation='relu'))
        self._LSTMmodel.add(tf.keras.layers.Dense(self._target_size))
        self._LSTMmodel.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')
        history = self._LSTMmodel.fit(tdat, tlab, epochs=20,batch_size=50,validation_data=(vdat, vlab))
        plot_train_history(history,'Single Step Training and validation loss')
        self._LSTMmodel.save('{}.h5'.format(name))

    def LSTMload(self, name):
        self._LSTMmodel = tf.keras.models.load_model('{}.h5'.format(name))

    def LSTMpredict(self, nbset, goto, **kwargs):
        pred=[]
        truth=[]
        if len(kwargs)==0:
            datas=np.array(self._val_datas)
            truth=self._val_labels[nbset:nbset+goto]
        elif len(kwargs)==2:
            datas=np.array(kwargs["datas"])
            truth=np.array(kwargs["labels"])[nbset:nbset+goto]
        else:
            print("wrong number of parameters - stopping")
            return
        for k in range(goto):
            # the input for the model
            position=nbset+k
            # make a deepcopy not to affect datas
            # cf http://lyceeomar.atspace.cc/Copiesup_profonde1.html
            sample=copy.deepcopy(datas[position])
            if self._debug:
                print("pred length : {}".format(len(pred)))
                print(pred)
            if len(pred)>0:
                if len(pred)>self._history_size:
                    # we want the last history_size predictions
                    simTs=pred[-self._history_size:]
                else:
                    simTs=pred
                if self._debug:
                    print("before injection")
                    print(sample)
                # we have to update the last min(len(pred),history_size) elements in the temperature column (1)
                sample[-min(len(pred),self._history_size):,1]=simTs
            if self._debug:
                print("after sim injection")
                print(sample)
                input("press any key")
            sample=sample.reshape(1,self._history_size,datas.shape[-2:][1])
            prediction=self._LSTMmodel.predict(sample)
            pred.append(prediction[0,0])
        if self._MLAregularize:
            pred=np.array(pred)*self._std[1]+self._mean[1]
            truth=np.array(truth)*self._std[1]+self._mean[1]
        return pred, truth

    # uses the val datas as train datas are shuffled
    # nbset is the number of the set in the val array
    # physics are the original unregularized datas
    def viewValSet(self, physics, nbset, nbpreds, preds=None):
        history_range=list(range(-self._history_size, 0))
        future_range=list(range(0,nbpreds))
        zero=nbset+self._history_size
        grnb=111
        ax1=plt.subplot(grnb)
        ax1.set_ylabel('°C')
        plt.title("sample nb {}\n cross -> dataset \n line -> timeseries".format(zero))
        plt.xlim([history_range[0], future_range[-1]])
        truefuture=self._val_labels[nbset]*self._std[1]+self._mean[1]
        plt.plot(0,truefuture, 'o', label='true future', color=self._col[1])
        if preds is not None:
            plt.plot(future_range,preds[0],'+',label="multilinear preds", color="black")
            plt.plot(future_range,preds[1],'o', color=self._col[1])
            plt.plot(future_range,preds[2],'*',label="LSTM preds", color="black")
            plt.plot(future_range,preds[3],'*',label="LSTM labels", color="orange")
        values=self._val_datas[nbset]*self._std+self._mean
        for k in range(3):
            if k == 2:
               ax1.tick_params(axis='y')
               plt.legend()
               ax2 = ax1.twinx()
               ax2.set_ylabel("Kwh")
            plt.plot(history_range,values[:,k], 'rx', color=self._col[k])
            # printing the initial dataset (nbset) with crosses
            plt.plot(history_range,physics[nbset:zero,k], color=self._col[k], label=self._lab[k])
            plt.plot(future_range,physics[zero:zero+nbpreds,k], color=self._col[k])
        ax2.tick_params(axis='y')
        plt.legend()
        plt.show()
