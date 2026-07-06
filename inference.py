import os
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import soundfile as sf

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

file_model = './logs_new_1117_3/ljs_base/G_914000.pth'
hps = utils.get_hparams_from_file('./configs/ljs_base.json')
device = torch.device("cpu")

net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model).cuda()
net_g.eval()

model, optimizer, learning_rate, epochs= utils.load_checkpoint(file_model, net_g, None)

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def vits(text, noise_scale, noise_scale_w, length_scale):

    text = text.replace('\n',' ').replace('\r',' ').replace(" ","")

    stn_tst = get_text(text, hps)
    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
        audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667,
                            noise_scale_w=0.8,
                            length_scale=1)[0][0,0].data.cpu().float().numpy()
    
    return 22050,audio

speed = 1
fs, audio = vits("welcome to Harbin", 0.1, 0.668, 1.0/speed)
sf.write('01.wav', audio, fs)
