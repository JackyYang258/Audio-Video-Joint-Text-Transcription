import itertools
import logging
import os
import sys
import time
from typing import Any, List, Optional, Union

import cv2
import numpy as np
from argparse import Namespace
import torch
import torch.nn.functional as F
from fairseq.data import data_utils, Dictionary, encoders
from fairseq.data.fairseq_dataset import FairseqDataset
from python_speech_features import logfbank
from scipy.io import wavfile

import utils

logger = logging.getLogger(__name__)


def load_audio_visual(manifest_path, max_keep, min_keep, frame_rate, label_paths, label_rates, tol=0.1):
    def is_audio_label_aligned(audio_dur, label_durs):
        return all([abs(audio_dur - label_dur)<tol for label_dur in label_durs])

    n_long, n_short, n_unaligned = 0, 0, 0
    names, inds, sizes = [], [], []
    dur_from_label_list = []
    is_seq_label = any([x==-1 for x in label_rates])
    for label_path, label_rate in zip(label_paths, label_rates):
        label_lengths = [len(line.rstrip().split())/label_rate for line in open(label_path).readlines()]
        dur_from_label_list.append(label_lengths)
    dur_from_label_list = list(zip(*dur_from_label_list))

    with open(manifest_path) as f:
        root = f.readline().strip()
        for ind, line in enumerate(f):
            items = line.strip().split("\t")
            sz = int(items[-2]) # 
            if min_keep is not None and sz < min_keep:
                n_short += 1
            elif max_keep is not None and sz > max_keep:
                n_long += 1
            elif (not is_seq_label) and (not is_audio_label_aligned(sz/frame_rate, dur_from_label_list[ind])):
                n_unaligned += 1
            else:
                video_path = items[1]
                audio_path = items[2]
                audio_id = items[0]
                names.append((video_path, audio_path+':'+audio_id))
                inds.append(ind)
                sizes.append(sz)
    tot = ind + 1
    logger.info(
        (
            f"max_keep={max_keep}, min_keep={min_keep}, "
            f"loaded {len(names)}, skipped {n_short} short and {n_long} long and {n_unaligned} unaligned, "
            f"longest-loaded={max(sizes)}, shortest-loaded={min(sizes)}"
        )
    )
    return root, names, inds, tot, sizes

def load_label(label_path, inds, tot):
    with open(label_path) as f:
        labels = [line.rstrip() for line in f]
        assert (
            len(labels) == tot
        ), f"number of labels does not match ({len(labels)} != {tot})"
        labels = [labels[i] for i in inds]
    return labels


def load_label_offset(label_path, inds, tot):
    with open(label_path) as f:
        code_lengths = [len(line.encode("utf-8")) for line in f]
        assert (
            len(code_lengths) == tot
        ), f"number of labels does not match ({len(code_lengths)} != {tot})"
        offsets = list(itertools.accumulate([0] + code_lengths))
        offsets = [(offsets[i], offsets[i + 1]) for i in inds]
    return offsets


def verify_label_lengths(
    audio_sizes,
    audio_rate,
    label_path,
    label_rate,
    inds,
    tot,
    tol=0.1,  # tolerance in seconds
):
    if label_rate < 0:
        logger.info(f"{label_path} is sequence label. skipped")
        return

    with open(label_path) as f:
        lengths = [len(line.rstrip().split()) for line in f]
        assert len(lengths) == tot
        lengths = [lengths[i] for i in inds]
    num_invalid = 0
    for i, ind in enumerate(inds):
        dur_from_audio = audio_sizes[i] / audio_rate
        dur_from_label = lengths[i] / label_rate
        if abs(dur_from_audio - dur_from_label) > tol:
            logger.warning(
                (
                    f"audio and label duration differ too much "
                    f"(|{dur_from_audio} - {dur_from_label}| > {tol}) "
                    f"in line {ind+1} of {label_path}. Check if `label_rate` "
                    f"is correctly set (currently {label_rate}). "
                    f"num. of samples = {audio_sizes[i]}; "
                    f"label length = {lengths[i]}"
                )
            )
            num_invalid += 1
    if num_invalid > 0:
        logger.warning(
            f"total {num_invalid} (audio, label) pairs with mismatched lengths"
        )


class AudioVideoDataset(FairseqDataset):
    def __init__(
            self,
            cfg,
            split,
            dictionary,
            shuffle = True,
    ):
        self.cfg = cfg

        # build label tokenizer
        logger.info(f"Using tokenizer")
        bpe_args = Namespace(**{'bpe': self.cfg.tokenizer_bpe_name, f"{self.cfg.tokenizer_bpe_name}_model": self.cfg.tokenizer_bpe_model})
        self.bpe_tokenizer = encoders.build_bpe(bpe_args)
        self.dictionary = dictionary

        # load meta-data for audio-visual inputs
        label_path = f"{self.get_label_dir()}/{split}.{self.cfg.labels[0]}"
        self.label_rate = self.cfg.label_rate
        manifest_path = f"{self.cfg.data}/{split}.tsv"
        self.audio_root, self.names, inds, tot, self.sizes = load_audio_visual(
            manifest_path, 
            self.cfg.max_sample_size, 
            self.cfg.min_sample_size, 
            frame_rate=self.cfg.sample_rate, 
            label_paths=[label_path], 
            label_rates=[self.label_rate]
        )

        # build data transformations
        image_aug = self.cfg.image_aug if split == 'train' else False
        if image_aug:
            self.transform = utils.Compose([
                utils.Normalize( 0.0,255.0 ),
                utils.RandomCrop((self.cfg.image_crop_size, self.cfg.image_crop_size)),
                utils.HorizontalFlip(0.5),
                utils.Normalize(self.cfg.image_mean, self.cfg.image_std) ])
        else:
            self.transform = utils.Compose([
                utils.Normalize( 0.0,255.0 ),
                utils.CenterCrop((self.cfg.image_crop_size, self.cfg.image_crop_size)),
                utils.Normalize(self.cfg.image_mean, self.cfg.image_std) ])
        logger.info(f"image transform: {self.transform}")

        self.modalities = set(self.cfg.modalities)
        self.sample_rate = self.cfg.sample_rate
        self.stack_order_audio = self.cfg.stack_order_audio
        self.shuffle = shuffle
        self.random_crop = self.cfg.random_crop

        self.single_target = self.cfg.single_target
        self.is_s2s = self.cfg.is_s2s
        self.pad_list = [dictionary.pad()]
        self.max_sample_size = (
            self.cfg.max_trim_sample_size if self.cfg.max_trim_sample_size is not None else sys.maxsize
        )
        self.pad_audio = self.cfg.pad_audio
        self.normalize = self.cfg.normalize # for audio

        self.label_path = label_path
        self.label_offsets_list = load_label_offset(label_path, inds, tot)

        logger.info(
            f"pad_audio={self.pad_audio}, random_crop={self.random_crop}, "
            f"normalize={self.normalize}, max_sample_size={self.max_sample_size}, "
            f"seqs2seq data={self.is_s2s},")

    def get_label_dir(self):
        if self.cfg.label_dir is None:
            return self.cfg.data
        return self.cfg.label_dir

    def get_label(self, index):
        label = None
        ################################################################################
        # TODO:                                                                        #
        #  1. read label string from label file                                        #
        #  2. split string into subwords with tokenizer                                #
        #  3. encoder each subword to long int                                         #
        ################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        #  1. read label string from label file 
        [s_start,s_end] = self.label_offsets_list[index]
        fp = open(self.label_path, 'r')
        fp.seek(s_start)
        sentence = fp.read(s_end - s_start)
        # print(sentence)

        #  2. split string into subwords with tokenizer  
        split_sentence = self.bpe_tokenizer.encode(sentence)

        #  3. encoder each subword to long int 
        split_sentence = split_sentence.split()
        label = [None] * len(split_sentence)
        for num in range(0, len(split_sentence)):
            label[num] = self.dictionary.index(split_sentence[num])
        label = np.array(label)
        label = torch.tensor(label)
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        return label

    def load_feature(self, mix_name):
        """
        Load image and audio feature
        Returns:
        video_feats: numpy.ndarray of shape [T, H, W, 1], audio_feats: numpy.ndarray of shape [T, F]
        """
        video_fn, audio_fn = mix_name
        if 'video' in self.modalities:
            video_feats = self.load_video(video_fn) # [T, H, W, 1]
        else:
            video_feats = None
        if 'audio' in self.modalities:
            audio_feats = self.load_audio(audio_fn.split(':')[0]) # [T, F]
        else:
            audio_feats = None
        if audio_feats is not None and video_feats is not None:
            diff = len(audio_feats) - len(video_feats)
            if diff < 0:
                audio_feats = np.concatenate([audio_feats, np.zeros([-diff, audio_feats.shape[-1]], dtype=audio_feats.dtype)])
            elif diff > 0:
                audio_feats = audio_feats[:-diff]
        return video_feats, audio_feats

    def load_video(self, video_name):
        feats = None
        ################################################################################
        # TODO:                                                                        #
        #  1. load video with cv2.VideoCapture                                         #
        #  2. apply self.transform                                                     #
        #  3. expand the feats shape to (T, H, W, 1)                                   #
        ################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        cap = cv2.VideoCapture(video_name)     # 读取视频
        print(cap.isOpened())  # 检查是否打开正确  

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        gray_cap = np.zeros((frame_count, frame_height, frame_width))

        for num in range(0, frame_count):
            ret, frame = cap.read()
            if frame is None:
                break
            if ret == True:
                gray_frame = cv2.cvtColor(frame,  cv2.COLOR_BGR2GRAY)
                gray_cap[num, :, :] = gray_frame

        gray_cap_T = self.transform(gray_cap)
        feats = np.expand_dims(gray_cap_T, axis=3)
        cap.release()

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        return feats
    
    def load_audio(self, audio_name):
        audio_feats = None
        ################################################################################
        # TODO:                                                                        #
        #  1. load audio with wavfile                                                  #
        #  2. extract feature with logfbank                                            #
        #  3. stack consecutive frames (number=self.stack_order_audio)                 #
        ################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################

        return audio_feats

    def select_noise(self):
        rand_indexes = np.random.randint(0, len(self.noise_wav), size=self.noise_num)
        noise_wav = []
        for x in rand_indexes:
            noise_wav.append(wavfile.read(self.noise_wav[x])[1].astype(np.float32))
        if self.noise_num == 1:
            return noise_wav[0]
        else:
            min_len = min([len(x) for x in noise_wav])
            noise_wav = [x[:min_len] for x in noise_wav]
            noise_wav = np.floor(np.stack(noise_wav).mean(axis=0))
            return noise_wav

    def add_noise(self, clean_wav):
        clean_wav = clean_wav.astype(np.float32)
        noise_wav = self.select_noise()
        if type(self.noise_snr) == int or type(self.noise_snr) == float:
            snr = self.noise_snr
        elif type(self.noise_snr) == tuple:
            snr = np.random.randint(self.noise_snr[0], self.noise_snr[1]+1)
        clean_rms = np.sqrt(np.mean(np.square(clean_wav), axis=-1))
        if len(clean_wav) > len(noise_wav):
            ratio = int(np.ceil(len(clean_wav)/len(noise_wav)))
            noise_wav = np.concatenate([noise_wav for _ in range(ratio)])
        if len(clean_wav) < len(noise_wav):
            start = 0
            noise_wav = noise_wav[start: start + len(clean_wav)]
        noise_rms = np.sqrt(np.mean(np.square(noise_wav), axis=-1))
        adjusted_noise_rms = clean_rms / (10**(snr/20))
        adjusted_noise_wav = noise_wav * (adjusted_noise_rms / noise_rms)
        mixed = clean_wav + adjusted_noise_wav

        #Avoid clipping noise
        max_int16 = np.iinfo(np.int16).max
        min_int16 = np.iinfo(np.int16).min
        if mixed.max(axis=0) > max_int16 or mixed.min(axis=0) < min_int16:
            if mixed.max(axis=0) >= abs(mixed.min(axis=0)): 
                reduction_rate = max_int16 / mixed.max(axis=0)
            else :
                reduction_rate = min_int16 / mixed.min(axis=0)
            mixed = mixed * (reduction_rate)
        mixed = mixed.astype(np.int16)
        return mixed

    def __getitem__(self, index):
        video_feats, audio_feats = self.load_feature(self.names[index])
        audio_feats = torch.from_numpy(audio_feats.astype(np.float32)) if audio_feats is not None else None
        video_feats = torch.from_numpy(video_feats.astype(np.float32)) if video_feats is not None else None
        if self.normalize and 'audio' in self.modalities:
            with torch.no_grad():
                audio_feats = F.layer_norm(audio_feats, audio_feats.shape[1:])
        labels = self.get_label(index)
        fid = self.names[index][1].split(':')[1]
        return {"id": index, 'fid': fid, "video_source": video_feats, 'audio_source': audio_feats, "label": labels}

    def __len__(self):
        return len(self.sizes)

    def crop_to_max_size(self, wav, target_size, start=None):
        size = len(wav)
        diff = size - target_size
        if diff <= 0:
            return wav, 0
        # longer utterances
        if start is None:
            start, end = 0, target_size
            if self.random_crop:
                start = np.random.randint(0, diff + 1)
                end = size - diff + start
        else:
            end = start + target_size
        return wav[start:end], start

    def collater(self, samples):
        samples = [s for s in samples if s["id"] is not None]
        if len(samples) == 0:
            return {}

        audio_source, video_source = [s["audio_source"] for s in samples], [s["video_source"] for s in samples]
        if audio_source[0] is None:
            audio_source = None
        if video_source[0] is None:
            video_source = None
        if audio_source is not None:
            audio_sizes = [len(s) for s in audio_source]
        else:
            audio_sizes = [len(s) for s in video_source]
        if self.pad_audio:
            audio_size = min(max(audio_sizes), self.max_sample_size)
        else:
            audio_size = min(min(audio_sizes), self.max_sample_size)
        if audio_source is not None:
            collated_audios, padding_mask, audio_starts = self.collater_audio(audio_source, audio_size)
        else:
            collated_audios, audio_starts = None, None
        if video_source is not None:
            collated_videos, padding_mask, audio_starts = self.collater_audio(video_source, audio_size, audio_starts)
        else:
            collated_videos = None
        targets_by_label = [
            [s["label"] for s in samples]
        ]
        targets_list, lengths_list, ntokens_list = self.collater_label(
            targets_by_label, audio_size, audio_starts
        )
        source = {"audio": collated_audios, "video": collated_videos}
        net_input = {"source": source, "padding_mask": padding_mask}
        batch = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": net_input,
            "utt_id": [s['fid'] for s in samples]
        }

        if self.single_target:
            batch["target_lengths"] = lengths_list[0]
            batch["ntokens"] = ntokens_list[0]
            if self.is_s2s:
                batch['target'], net_input['prev_output_tokens'] = targets_list[0][0], targets_list[0][1]
            else:
                batch["target"] = targets_list[0]
        else:
            batch["target_lengths_list"] = lengths_list
            batch["ntokens_list"] = ntokens_list
            batch["target_list"] = targets_list
        return batch

    def collater_audio(self, audios, audio_size, audio_starts=None):
        audio_feat_shape = list(audios[0].shape[1:])
        collated_audios = audios[0].new_zeros([len(audios), audio_size]+audio_feat_shape)
        padding_mask = (
            torch.BoolTensor(len(audios), audio_size).fill_(False) # 
        )
        start_known = audio_starts is not None
        audio_starts = [0 for _ in audios] if not start_known else audio_starts
        for i, audio in enumerate(audios):
            diff = len(audio) - audio_size
            if diff == 0:
                collated_audios[i] = audio
            elif diff < 0:
                assert self.pad_audio
                collated_audios[i] = torch.cat(
                    [audio, audio.new_full([-diff]+audio_feat_shape, 0.0)]
                )
                padding_mask[i, diff:] = True
            else:
                collated_audios[i], audio_starts[i] = self.crop_to_max_size(
                    audio, audio_size, audio_starts[i] if start_known else None
                )
        if len(audios[0].shape) == 2:
            collated_audios = collated_audios.transpose(1, 2) # [B, T, F] -> [B, F, T]
        else:
            collated_audios = collated_audios.permute((0, 4, 1, 2, 3)).contiguous() # [B, T, H, W, C] -> [B, C, T, H, W]
        return collated_audios, padding_mask, audio_starts

    def collater_frm_label(
        self, targets, audio_size, audio_starts, label_rate, pad
    ):
        assert label_rate > 0
        s2f = label_rate / self.sample_rate # num label per sample
        frm_starts = [int(round(s * s2f)) for s in audio_starts]
        frm_size = int(round(audio_size * s2f))
        if not self.pad_audio:
            rem_size = [len(t) - s for t, s in zip(targets, frm_starts)]
            frm_size = min(frm_size, *rem_size)
        targets = [t[s: s + frm_size] for t, s in zip(targets, frm_starts)]
        logger.debug(f"audio_starts={audio_starts}")
        logger.debug(f"frame_starts={frm_starts}")
        logger.debug(f"frame_size={frm_size}")

        lengths = torch.LongTensor([len(t) for t in targets])
        ntokens = lengths.sum().item()
        targets = data_utils.collate_tokens(
            targets, pad_idx=pad, left_pad=False
        )
        return targets, lengths, ntokens

    def collater_seq_label(self, targets, pad):
        lengths = torch.LongTensor([len(t) for t in targets])
        ntokens = lengths.sum().item()
        targets = data_utils.collate_tokens(
            targets, pad_idx=pad, left_pad=False
        )
        return targets, lengths, ntokens

    def collater_seq_label_s2s(self, targets, pad):
        lengths = torch.LongTensor([len(t) for t in targets])
        ntokens = lengths.sum().item()
        pad, eos = self.dictionary.pad(), self.dictionary.eos()
        targets_ = data_utils.collate_tokens(targets, pad_idx=pad, eos_idx=eos, left_pad=False)
        prev_output_tokens = data_utils.collate_tokens(targets, pad_idx=pad, eos_idx=eos, left_pad=False, move_eos_to_beginning=True)
        return (targets_, prev_output_tokens), lengths, ntokens

    def collater_label(self, targets_by_label, audio_size, audio_starts):
        targets_list, lengths_list, ntokens_list = [], [], []
        itr = zip(targets_by_label, [self.label_rate], self.pad_list)
        for targets, label_rate, pad in itr:
            if label_rate == -1:
                if self.is_s2s:
                    targets, lengths, ntokens = self.collater_seq_label_s2s(targets, pad)
                else:
                    targets, lengths, ntokens = self.collater_seq_label(targets, pad)
            else:
                targets, lengths, ntokens = self.collater_frm_label(
                    targets, audio_size, audio_starts, label_rate, pad
                )
            targets_list.append(targets)
            lengths_list.append(lengths)
            ntokens_list.append(ntokens)
        return targets_list, lengths_list, ntokens_list

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        if self.pad_audio:
            return self.sizes[index]
        return min(self.sizes[index], self.max_sample_size)

    def ordered_indices(self):
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        order.append(self.sizes)
        return np.lexsort(order)[::-1]
