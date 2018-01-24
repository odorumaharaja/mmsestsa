#! /usr/bin/env python
# -*- coding: utf-8 -*-

'''
Short time Spectral Amplitude Minimum Mean Square Error Method for Denoising noisy speech.
'''

import sys
import argparse

import numpy as np
import scipy as sp
import scipy.signal
import scipy.special as spc

# pip install pysoundfile
import soundfile as sf

def MMSESTSA(signal, fs, IS=0.25, W=1024, NoiseMargin=3, saved_params=None):
    '''
    MMSE-STSA method

    Args:
        signal : 一次元の入力信号
        fs     : サンプリング周波数
        IS     : 初期化用の無音 [sec] (default: 0.25)
    '''

    # window length is 25 msec
    #W = np.fix( 0.25 * fs )
    
    # Shift percentage is 40% (=10msec)
    # Overlap-Add method works good with this value(.4)
    SP = 0.4

    wnd = np.hamming(W)

    # pre-emphasis
    pre_emph = 0
    signal = scipy.signal.lfilter([1 -pre_emph],1,signal)

    # number of initial silence segments
    NIS = int(np.fix((IS*fs-W)/(SP*W) + 1))

    # This function chops the signal into frames
    y = segment(signal, W, SP, wnd)
    Y = np.fft.fft(y, axis=0)

    # Noisy Speech Phase
    YPhase = np.angle(Y[0:int(np.fix(len(Y)/2))+1,:])

    # Spectrogram
    Y = np.abs(Y[0:int(np.fix(len(Y)/2))+1,:])

    numberOfFrames = Y.shape[1]

    # initial Noise Power Spectrum mean
    N = np.mean(Y[:,0:NIS].T).T

    # initial Noise Power Spectrum variance
    LambdaD = np.mean((Y[:,0:NIS].T) ** 2).T

    # used in smoothing xi (For Deciesion Directed method for estimation of A Priori SNR)
    alpha = 0.99

    # This is a smoothing factor for the noise updating
    NoiseLength = 9
    NoiseCounter = 0

    if saved_params != None:
        NIS = 0
        N = saved_params['N']
        LambdaD = saved_params['LambdaD']
        NoiseCounter = saved_params['NoiseCounter']

    # Initial Gain used in calculation of the new xi
    G = np.ones(N.shape)
    Gamma = G

    # Gamma function at 1.5
    Gamma1p5 = spc.gamma(1.5)
    X = np.zeros(Y.shape)

    for i in range(numberOfFrames):
        Y_i = Y[:,i]

        # If initial silence ignore VAD
        if i < NIS:
            SpeechFlag = 0
            NoiseCounter = 100
        else:
            # %Magnitude Spectrum Distance VAD
            NoiseFlag, SpeechFlag, NoiseCounter = vad(Y_i, N, NoiseCounter, NoiseMargin)

        # If not Speech Update Noise Parameters
        if SpeechFlag == 0:
            N = (NoiseLength * N + Y_i) / (NoiseLength + 1)
            LambdaD = (NoiseLength * LambdaD + (Y_i ** 2)) / (1 + NoiseLength)

        # A postiriori SNR
        gammaNew = (Y_i ** 2) / LambdaD
        # Decision Directed Method for A Priori SNR
        xi = alpha * (G ** 2) * Gamma + (1 - alpha) * np.maximum(gammaNew - 1, 0)

        Gamma = gammaNew
        
        # A Function used in Calculation of Gain
        nu = Gamma * xi / (1 + xi)

        # MMSE STSA algo
        G = (Gamma1p5 * np.sqrt(nu) / Gamma) * np.exp(-nu / 2.0) *\
             ((1.0 + nu) * spc.i0(nu / 2.0) + nu * spc.i1(nu / 2.0))
        Indx = np.isnan(G) | np.isinf(G)
        G[Indx] = xi[Indx] / (1 + xi[Indx])

        X[:,i] = G * Y_i

    output = OverlapAdd2(X, YPhase, W, SP * W)
    return output, {'N': N, 'LambdaD': LambdaD, 'NoiseCounter': NoiseCounter}

def OverlapAdd2(XNEW, yphase, windowLen, ShiftLen):
    '''
    スペクトログラムから復元された信号を返す

    Args:
        XNEW : 音声のセグメントをFFTしたもの
        yphase : XNEWの位相成分
        windowLen : 窓幅
        ShiftLen : シフト幅

    Returns
        復元信号
    '''
    
    FrameNum = XNEW.shape[1]
    Spec = XNEW * np.exp(1j * yphase)

    ShiftLen = int(np.fix(ShiftLen))

    if windowLen % 2:
        Spec = np.concatenate((Spec, np.flipud(np.conj(Spec[1:,]))))
    else:
        Spec = np.concatenate((Spec, np.flipud(np.conj(Spec[1:-1,:]))))

    sig = np.zeros(((FrameNum - 1) * ShiftLen + windowLen, 1)) 

    for i in range(FrameNum):
        start = i * ShiftLen
        spec = Spec[:,[i]]
        sig[start:start + windowLen] = sig[start:start + windowLen] + np.real(np.fft.ifft(spec, axis=0))

    return sig

def segment(signal, W=256, SP=0.4, Window=None):
    '''
    音声信号をオーバーラップして切り出す
    出力は窓関数をかけた１次元のフレーム

    Args:
        signal : 入力信号
        W : 窓幅 (default: 256)
        SP : シフト割合 (defualt: 0.4)
        Windows : 窓関数
    Returns:
        １次元のフレーム
    '''
    if Window is None:
        Window = np.hamming(W)

    # make it a column vector
    Window = Window.flatten()

    L = len(signal)
    SP = int(np.fix(W * SP))

    # number of segments
    N = int(np.fix((L-W)/SP +1))

    Index = (np.tile(np.arange(0,W), (N,1)) + np.tile(np.arange(0,N) * SP, (W,1)).T).T
    hw = np.tile(Window, (N, 1)).T

    Seg = signal[Index] * hw
    
    return Seg

def vad(signal, noise, NoiseCounter = 0, NoiseMargin = 3, Hangover = 8):
    '''
    Spectral Distance Voice Activity Detector

    Args:
        signal : 現在のフレームの振幅成分（ノイズ or 音声のラベル付き）
        noise  : ノイズの振幅成分
        NoiseCounter : 直前のノイズフレームの数
        NoiseMargin : スペクトル距離(default: 3)
        Hangover : Speech Flagのリセットに使うフレーム数(default: 8)

    Returns:
        NoiseFlag : ノイズの場合は1を返す
        SpeechFlag : 音声の場合は1を返す
        NoiseCounter : 直前のノイズフレームの数
    '''
    SpectralDist = 20 * (np.log10(signal) - np.log10(noise))
    SpectralDist[SpectralDist < 0] = 0

    Dist = np.mean(SpectralDist)
    
    if (Dist < NoiseMargin):
        NoiseFlag = 1
        NoiseCounter = NoiseCounter + 1
    else:
        NoiseFlag = 0
        NoiseCounter = 0
    
    # Detect noise only periods and attenuate the signal     
    if (NoiseCounter > Hangover):
        SpeechFlag=0
    else:
        SpeechFlag=1

    return NoiseFlag, SpeechFlag, NoiseCounter

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Speech enhancement/noise reduction using Log MMSE STSA algorithm')
    parser.add_argument('input_file', action='store', type=str, help='input file to clean')
    parser.add_argument('output_file', action='store', type=str, help='output file to write (default: stdout)', default=sys.stdout)
    parser.add_argument('-i, --initial-noise', action='store', type=float, dest='initial_noise', help='initial noise in ms (default: 0.1)', default=0.1)
    parser.add_argument('-w, --window-size', action='store', type=int, dest='window_size', help='hamming window size (default: 1024)', default=1024)
    parser.add_argument('-n, --noise-threshold', action='store', type=int, dest='noise_threshold', help='noise thresold (default: 3)', default=3)
    args = parser.parse_args()

    data, fs = sf.read(args.input_file)
    num_frames = len(data)

    chunk_size = int(np.fix(60*fs))
    saved_params = None

    reconst = None

    frames_read = 0
    while (frames_read < num_frames):
        frames = num_frames - frames_read if frames_read + chunk_size > num_frames else chunk_size
        signal = data[frames_read:frames_read+frames]
        frames_read = frames_read + frames

        output, saved_params = MMSESTSA(signal, fs, args.initial_noise, args.window_size, args.noise_threshold, saved_params)

        if reconst is None:
            reconst = output
        else:
            reconst = np.hstack((reconst, output))

    sf.write('new_file.wav', reconst, fs)
