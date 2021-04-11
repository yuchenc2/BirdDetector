import json

import numpy as np
import pandas as pd
from fastprogress import progress_bar
import librosa
import torchvision.models as models
import torch
import warnings
from collections import defaultdict
from collections import Counter

list_of_models = [
    {
        "model_class": PANNsDense121Att,
        "config": {
            "sample_rate": 32000,
            "window_size": 1024,
            "hop_size": 320,
            "mel_bins": 64,
            "fmin": 50,
            "fmax": 14000,
            "classes_num": 264,
            "apply_aug": True,
            "top_db": None
        },
        "weights_path": "../input/birdsongdetectionfinalsubmission1/final_sed_dense121_nomix_fold0_checkpoint_50_score0.7057.pt",
        "clip_threshold": 0.3,
        "threshold": 0.3
    },
    {
        "model_class": PANNsDense121Att,
        "config": {
            "sample_rate": 32000,
            "window_size": 1024,
            "hop_size": 320,
            "mel_bins": 64,
            "fmin": 50,
            "fmax": 14000,
            "classes_num": 264,
            "apply_aug": True,
            "top_db": None
        },
        "weights_path": "../input/birdsongdetectionfinalsubmission1/final_sed_dense121_nomix_fold1_checkpoint_48_score0.6943.pt",
        "clip_threshold": 0.3,
        "threshold": 0.3
    },
    {
        "model_class": PANNsDense121Att,
        "config": {
            "sample_rate": 32000,
            "window_size": 1024,
            "hop_size": 320,
            "mel_bins": 64,
            "fmin": 50,
            "fmax": 14000,
            "classes_num": 264,
            "apply_aug": True,
            "top_db": None
        },
        "weights_path": "../input/birdsongdetectionfinalsubmission1/final_sed_dense121_nomix_fold2_augd_checkpoint_50_score0.6666.pt",
        "clip_threshold": 0.3,
        "threshold": 0.3
    },
    {
        "model_class": PANNsDense121Att,
        "config": {
            "sample_rate": 32000,
            "window_size": 1024,
            "hop_size": 320,
            "mel_bins": 64,
            "fmin": 50,
            "fmax": 14000,
            "classes_num": 264,
            "apply_aug": True,
            "top_db": None
        },
        "weights_path": "../input/birdsongdetectionfinalsubmission1/final_sed_dense121_nomix_fold3_augd_checkpoint_50_score0.6713.pt",
        "clip_threshold": 0.3,
        "threshold": 0.3
    },
    {
        "model_class": PANNsDense121Att,
        "config": {
            "sample_rate": 32000,
            "window_size": 1024,
            "hop_size": 320,
            "mel_bins": 64,
            "fmin": 50,
            "fmax": 14000,
            "classes_num": 264,
            "apply_aug": True,
            "top_db": None
        },
        "weights_path": "../input/birdsongdetectionfinalsubmission1/final_5fold_sed_dense121_nomix_fold0_checkpoint_50_score0.7219.pt",
        "clip_threshold": 0.3,
        "threshold": 0.3
    },
    {
        "model_class": PANNsDense121Att,
        "config": {
            "sample_rate": 32000,
            "window_size": 1024,
            "hop_size": 320,
            "mel_bins": 64,
            "fmin": 50,
            "fmax": 14000,
            "classes_num": 264,
            "apply_aug": True,
            "top_db": None
        },
        "weights_path": "../input/birdsongdetectionfinalsubmission1/final_5fold_sed_dense121_nomix_fold1_checkpoint_44_score0.7645.pt",
        "clip_threshold": 0.3,
        "threshold": 0.3
    },
    {
        "model_class": PANNsDense121Att,
        "config": {
            "sample_rate": 32000,
            "window_size": 1024,
            "hop_size": 320,
            "mel_bins": 64,
            "fmin": 50,
            "fmax": 14000,
            "classes_num": 264,
            "apply_aug": True,
            "top_db": None
        },
        "weights_path": "../input/birdsongdetectionfinalsubmission1/final_5fold_sed_dense121_nomix_fold2_checkpoint_50_score0.7737.pt",
        "clip_threshold": 0.3,
        "threshold": 0.3
    },
    {
        "model_class": PANNsDense121Att,
        "config": {
            "sample_rate": 32000,
            "window_size": 1024,
            "hop_size": 320,
            "mel_bins": 64,
            "fmin": 50,
            "fmax": 14000,
            "classes_num": 264,
            "apply_aug": True,
            "top_db": None
        },
        "weights_path": "../input/birdsongdetectionfinalsubmission1/final_5fold_sed_dense121_nomix_fold3_checkpoint_48_score0.7746.pt",
        "clip_threshold": 0.3,
        "threshold": 0.3
    },
    {
        "model_class": PANNsDense121Att,
        "config": {
            "sample_rate": 32000,
            "window_size": 1024,
            "hop_size": 320,
            "mel_bins": 64,
            "fmin": 50,
            "fmax": 14000,
            "classes_num": 264,
            "apply_aug": True,
            "top_db": None
        },
        "weights_path": "../input/birdsongdetectionfinalsubmission1/final_5fold_sed_dense121_nomix_fold4_checkpoint_50_score0.7728.pt",
        "clip_threshold": 0.3,
        "threshold": 0.3
    },
    {
        "model_class": PANNsDense121Att,
        "config": {
            "sample_rate": 32000,
            "window_size": 1024,
            "hop_size": 320,
            "mel_bins": 64,
            "fmin": 50,
            "fmax": 14000,
            "classes_num": 264,
            "apply_aug": True,
            "top_db": None
        },
        "weights_path": "../input/birdsongdetectionfinalsubmission1/final_sed_dense121_mix_fold0_2_checkpoint_50_score0.6842.pt",
        "clip_threshold": 0.3,
        "threshold": 0.3
    },
    {
        "model_class": PANNsDense121Att,
        "config": {
            "sample_rate": 32000,
            "window_size": 1024,
            "hop_size": 320,
            "mel_bins": 64,
            "fmin": 50,
            "fmax": 14000,
            "classes_num": 264,
            "apply_aug": True,
            "top_db": None
        },
        "weights_path": "../input/birdsongdetectionfinalsubmission1/final_sed_dense121_mix_fold1_2_checkpoint_50_score0.6629.pt",
        "clip_threshold": 0.3,
        "threshold": 0.3
    },
    {
        "model_class": PANNsDense121Att,
        "config": {
            "sample_rate": 32000,
            "window_size": 1024,
            "hop_size": 320,
            "mel_bins": 64,
            "fmin": 50,
            "fmax": 14000,
            "classes_num": 264,
            "apply_aug": True,
            "top_db": None
        },
        "weights_path": "../input/birdsongdetectionfinalsubmission1/final_sed_dense121_mix_fold2_2_checkpoint_50_score0.6884.pt",
        "clip_threshold": 0.3,
        "threshold": 0.3
    },
    {
        "model_class": PANNsDense121Att,
        "config": {
            "sample_rate": 32000,
            "window_size": 1024,
            "hop_size": 320,
            "mel_bins": 64,
            "fmin": 50,
            "fmax": 14000,
            "classes_num": 264,
            "apply_aug": True,
            "top_db": None
        },
        "weights_path": "../input/birdsongdetectionfinalsubmission1/final_sed_dense121_mix_fold3_2_checkpoint_50_score0.6870.pt",
        "clip_threshold": 0.3,
        "threshold": 0.3
    }
]
PERIOD = 30
SR = 32000
vote_lim = 4
TTA = 10

def get_model(ModelClass: object, config: dict, weights_path: str):
    model = ModelClass(**config)
    checkpoint = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model
    
def prediction_for_clip(test_df: pd.DataFrame,
                        clip: np.ndarray, 
                        model,
                        threshold,
                       clip_threshold):

    audios = []
    y = clip.astype(np.float32)
    len_y = len(y)
    start = 0
    end = PERIOD * SR
    while True:
        y_batch = y[start:end].astype(np.float32)
        if len(y_batch) != PERIOD * SR:
            y_pad = np.zeros(PERIOD * SR, dtype=np.float32)
            y_pad[:len(y_batch)] = y_batch
            audios.append(y_pad)
            break
        start = end
        end += PERIOD * SR
        audios.append(y_batch)
        
    array = np.asarray(audios)
    tensors = torch.from_numpy(array)
    
    model.eval()
    estimated_event_list = []
    global_time = 0.0
    site = test_df["site"].values[0]
    audio_id = test_df["audio_id"].values[0]
    for image in tensors:
        image = image.unsqueeze(0).unsqueeze(0)
        image = image.expand(image.shape[0], TTA, image.shape[2])
        image = image.to(device)
        
        with torch.no_grad():
            prediction = model((image, None))
            framewise_outputs = prediction["framewise_output"].detach(
                ).cpu().numpy()[0].mean(axis=0)
            clipwise_outputs = prediction["clipwise_output"].detach(
                ).cpu().numpy()[0].mean(axis=0)
                
        thresholded = framewise_outputs >= threshold
        
        clip_thresholded = clipwise_outputs >= clip_threshold
        clip_indices = np.argwhere(clip_thresholded).reshape(-1)
        clip_codes = []
        for ci in clip_indices:
            clip_codes.append(INV_BIRD_CODE[ci])
            
        for target_idx in range(thresholded.shape[1]):
            if thresholded[:, target_idx].mean() == 0:
                pass
            else:
                detected = np.argwhere(thresholded[:, target_idx]).reshape(-1)
                head_idx = 0
                tail_idx = 0
                while True:
                    if (tail_idx + 1 == len(detected)) or (
                            detected[tail_idx + 1] - 
                            detected[tail_idx] != 1):
                        onset = 0.01 * detected[
                            head_idx] + global_time
                        offset = 0.01 * detected[
                            tail_idx] + global_time
                        onset_idx = detected[head_idx]
                        offset_idx = detected[tail_idx]
                        max_confidence = framewise_outputs[
                            onset_idx:offset_idx, target_idx].max()
                        mean_confidence = framewise_outputs[
                            onset_idx:offset_idx, target_idx].mean()
                        if INV_BIRD_CODE[target_idx] in clip_codes:
                            estimated_event = {
                                "site": site,
                                "audio_id": audio_id,
                                "ebird_code": INV_BIRD_CODE[target_idx],
                                "clip_codes": clip_codes,
                                "onset": onset,
                                "offset": offset,
                                "max_confidence": max_confidence,
                                "mean_confidence": mean_confidence
                            }
                            estimated_event_list.append(estimated_event)
                        head_idx = tail_idx + 1
                        tail_idx = tail_idx + 1
                        if head_idx >= len(detected):
                            break
                    else:
                        tail_idx += 1
        global_time += PERIOD
        
    prediction_df = pd.DataFrame(estimated_event_list)
    return prediction_df

def prediction(test_df: pd.DataFrame,
               test_audio,
               list_of_model_details):
    unique_audio_id = test_df.audio_id.unique()

    warnings.filterwarnings("ignore")
    prediction_dfs_dict = defaultdict(list)
    for audio_id in progress_bar(unique_audio_id):
        clip, _ = librosa.load(test_audio,
                               sr=SR,
                               mono=True,
                               res_type="kaiser_fast")
        
        test_df_for_audio_id = test_df.query(
            f"audio_id == '{audio_id}'").reset_index(drop=True)
        for i, model_details in enumerate(list_of_model_details):
            prediction_df = prediction_for_clip(test_df_for_audio_id,
                                                clip=clip,
                                                model=model_details["model"],
                                                threshold=model_details["threshold"],
                                               clip_threshold=model_details["clip_threshold"])

            prediction_dfs_dict[i].append(prediction_df)
    list_of_prediction_df = []
    for key, prediction_dfs in prediction_dfs_dict.items():
        prediction_df = pd.concat(prediction_dfs, axis=0, sort=False).reset_index(drop=True)
        list_of_prediction_df.append(prediction_df)
    return list_of_prediction_df
    
def get_post_post_process_predictions(prediction_df):
    labels = {}

    for audio_id, sub_df in progress_bar(prediction_df.groupby("audio_id")):
        events = sub_df[["ebird_code", "onset", "offset", "max_confidence", "site"]].values
        n_events = len(events)

        site = events[0][4]
        for i in range(n_events):
            event = events[i][0]
            onset = events[i][1]
            offset = events[i][2]
            
            start_section = int((onset // 5) * 5) + 5
            end_section = int((offset // 5) * 5) + 5
            cur_section = start_section

            row_id = f"{site}_{audio_id}_{start_section}"
            if labels.get(row_id) is not None:
                labels[row_id].add(event)
            else:
                labels[row_id] = set()
                labels[row_id].add(event)

            while cur_section != end_section:
                cur_section += 5
                row_id = f"{site}_{audio_id}_{cur_section}"
                if labels.get(row_id) is not None:
                    labels[row_id].add(event)
                else:
                    labels[row_id] = set()
                    labels[row_id].add(event)


    for key in labels:
        labels[key] = " ".join(sorted(list(labels[key])))


    row_ids = list(labels.keys())
    birds = list(labels.values())
    post_processed = pd.DataFrame({
        "row_id": row_ids,
        "birds": birds
    })
    return post_processed

if __name__ == "__main__":
    tdf = pd.DataFrame({
        "site": [1]
        "row_id": [1],
        "seconds": [1],
        "audio_id": [1]
    })
    list_of_prediction_df = prediction(test_df=tdf,
                           test_audio="XC134874.mp3",
                           list_of_model_details=list_of_models)
                           
    all_row_id = test[["row_id"]]
    list_of_submissions = []
    for prediction_df in list_of_prediction_df:
        post_processed = get_post_post_process_predictions(prediction_df)
        submission = post_processed.fillna("nocall")
        submission = submission.set_index('row_id')
        list_of_submissions.append(submission)
    
    list_all_of_row_ids = []
    for sub_x in list_of_submissions:
        list_all_of_row_ids+= list(sub_x.index.values)
    list_all_of_row_ids = list(set(list_all_of_row_ids))
    
    final_submission = []
    for row_id in list_all_of_row_ids:
        birds = []
        for sub in list_of_submissions:
            if row_id in sub.index:
                birds.extend(sub.loc[row_id].birds.split(" "))
        birds = [x for x in birds if "nocall" != x and "" != x]
        count_birds = Counter(birds)
        final_birds = []
        for key, value in count_birds.items():
            if value >= vote_lim:
                final_birds.append(key)
        if len(final_birds)>0:
            row_data = {
                "row_id": row_id,
                "birds": " ".join(sorted(final_birds))
            }
        else:
            row_data = {
                "row_id": row_id,
                "birds": "nocall"
            }
        final_submission.append(row_data)
        
    site_3_data = defaultdict(list)
    for row in final_submission:
        if "site_3" in row["row_id"]:
            final_row_id = "_".join(row["row_id"].split("_")[0:-1])
            birds = row["birds"].split(" ")
            birds = [x for x in birds if "nocall" != x and "" != x]
            site_3_data[final_row_id].extend(birds)
            
    for key, value in site_3_data.items():
        count_birds = Counter(value)
        final_birds = []
        for k, v in count_birds.items():
            if v >= vote_lim:
                final_birds.append(k)
        if len(final_birds)>0:
            row_data = {
                "row_id": key,
                "birds": " ".join(sorted(final_birds))
            }
        else:
            row_data = {
                "row_id": key,
                "birds": "nocall"
            }
        final_submission.append(row_data)
    
    final_submission = pd.DataFrame(final_submission)
    final_submission = all_row_id.merge(final_submission, on="row_id", how="left")
    final_submission = final_submission.fillna("nocall")
    print(final_submission['birds'].iloc(0))