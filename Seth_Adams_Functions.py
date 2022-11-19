#External deps
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Functions:
    def __init__(self):
        None

    def plot_signal(self, signal):
        plt.plot(signal)
        plt.suptitle('Time Series', size=16)

    def plot_signals(self, signals, nrows=2):
    #plot vars
        ncols = len(signals.keys())//nrows

        fig, axes = plt.subplots(nrows=nrows,ncols=ncols,sharex=False,
        sharey = True, figsize=(20,5))
        fig.suptitle('Time Series', size=16)
        i = 0
        for x in range(nrows):
            for y in range(ncols):
                axes[x,y].set_title(list(signals.keys())[i])
                axes[x,y].plot(list(signals.values())[i])
                axes[x,y].get_xaxis().set_visible(False)
                axes[x,y].get_yaxis().set_visible(False)
                i+=1

    def plot_fft(self, fft, nrows=2):
    #plot vars
        ncols = len(fft.keys())//nrows

        fig, axes = plt.subplots(nrows=nrows,ncols=ncols,sharex=False,
        sharey = False, figsize=(20,nrows*3.5))
        fig.suptitle('Fourier Transforms', size=16)
        i = 0
        for x in range(nrows):
            for y in range(ncols):
                data = list(fft.values())[i]
                Y, freq = data[0], data[1]
                axes[x,y].set_xscale('log')
                axes[x,y].set_title(list(fft.keys())[i])
                axes[x,y].plot(freq, Y)
                axes[x,y].get_xaxis().set_visible(True)
                axes[x,y].get_yaxis().set_visible(True)
                i+=1

    def plot_fbank(self, fbank):
        nrows = 2
        ncols = len(fbank.keys())//nrows

        fig, axes = plt.subplots(nrows=nrows,ncols=ncols,sharex=False,
        sharey = True, figsize=(20,5))
        fig.suptitle('Filter Bank Coefficients', size=16)
        i = 0
        for x in range(nrows):
            for y in range(ncols):
                axes[x,y].set_title(list(fbank.keys())[i])
                axes[x,y].imshow(list(fbank.values())[i],
                cmap='hot', interpolation='nearest')
                axes[x,y].get_xaxis().set_visible(False)
                axes[x,y].get_yaxis().set_visible(False)
                i+=1

    def plot_mfccs(self, mfccs):
        nrows = 2
        ncols = len(mfccs.keys())//nrows

        fig, axes = plt.subplots(nrows=nrows,ncols=ncols,sharex=False,
        sharey = True, figsize=(20,5))
        fig.suptitle('Mel Frequency Cepstrum Coefficients', size=16)
        i = 0
        for x in range(nrows):
            for y in range(ncols):
                axes[x,y].set_title(list(mfccs.keys())[i])
                axes[x,y].imshow(list(mfccs.values())[i],
                cmap='hot', interpolation='nearest')
                axes[x,y].get_xaxis().set_visible(False)
                axes[x,y].get_yaxis().set_visible(False)
                i+=1

    def envelope(self, y,rate,threshold):
        mask = [] #mask our data so that if the signal dies out and becomes irrelevant we ignore it
        y = pd.Series(y).apply(np.abs)
        y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True).mean()
        for mean in y_mean:
            if mean > threshold:
                mask.append(True)
            else:
                mask.append(False)
        return mask
    
    def calc_fft(self, y, rate):
        y -= np.mean(y)
        n = len(y)
        freq = np.fft.rfftfreq(n, d=1/rate)
        Y = abs(np.fft.rfft(y)/n)
        return (Y, freq)

    def calc_fft_over20(self, y, rate):
            n = len(y)
            freq = np.fft.rfftfreq(n, d=1/rate)
            Y = abs(np.fft.rfft(y)/n)
            return (Y[20:], freq[20:])