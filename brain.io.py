

import scipy.io
from io import BytesIO
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from mne.decoding import CSP





def main():
    mat = scipy.io.loadmat('ecog/Walk.mat')
    ecog_data = mat['y']

    labels = {
        'digit': ['0:10-0:17', '0:36-0:41', '3:51-3:54', '3:59-4:02'],
        'kanji': ['0:27-0:36', '3:42-3:46', '3:54-3:56', '4:02-4:04'],
        'face': ['0:47-0:53', '1:53-2:00', '2:31-2:37', '2:46-2:52', '2:59-3:04', '4:05-4:12'],
        'hira': ['0:56-1:02'],
        "object": ['1:25-1:32', '1:53-1:55', '1:59-2:03', '2:10-2:20', '3:12-3:30', '3:48-4:04'],
        "line": ['1:35-1:40', '3:49-3:51'],
        "body": ['1:48-2:03', '2:26-2:36', '2:53-3:03', '3:32-3:45', '3:56-3:59']
    }

    label_channels = np.zeros((7, 322049))

    for key in labels:

        key_to_idx = {
            'body': 0,
            'face': 1,
            'digit': 2,
            'hira': 3,
            'kanji': 4,
            'line': 5,
            'object': 6,
        }

        for span in labels[key]:
            start_time, end_time = span.split('-')
            min, sec = start_time.split(':')
            start_frame = (int(min) * 60 + int(sec) * 1) * 1200
            min, sec = end_time.split(':')
            end_frame = (int(min) * 60 + int(sec) * 1) * 1200
            # print(key, start_frame, end_frame)
            label_channels[key_to_idx[key]][start_frame:end_frame] = 1

    # get the indices of the columns where the last row is 1
    video_indices = np.where(ecog_data[-1] == 1)[0]
    video_data = ecog_data[:, video_indices]
    stim_1_data = ecog_data[:, np.where(ecog_data[-2] == 1)[0]]
    stim_2_data = ecog_data[:, np.where(ecog_data[-2] == 2)[0]]
    stim_3_data = ecog_data[:, np.where(ecog_data[-2] == 3)[0]]

    x = video_data[1:161]

    # Implement bandpass filter to isolate broadband gamma activity
    fs = 1200

    # 110-140
    low_pass_freq = 110
    high_pass_freq = 140
    filter_order = 4
    nyq_freq = 0.5 * fs

    b, a = signal.butter(filter_order, [low_pass_freq / nyq_freq, high_pass_freq / nyq_freq], btype='bandpass')

    x_filt = signal.filtfilt(b, a, x, axis=1)

    num_classes = len(np.unique(labels))

    csp_list = []
    lda_list = []

    for class_number in range(num_classes):
        labels_binary = np.zeros(labels.shape)
        labels_binary[labels == class_number] = 1
        labels_binary[labels != class_number] = -1

        # Fit pipeline to data
        csp = CSP(n_components=4, reg=None, log=None, norm_trace=False)
        lda = LDA()
        clf = Pipeline([('CSP', csp), ('LDA', lda)])
        clf.fit(prepped_data, labels_binary)

        # Save for later
        csp_list.append(csp)
        lda_list.append(lda)



    labels = np.zeros(shape = (ecog_data.shape[0],2))
    print(labels)

    labels = np.array([labels])
    ecog_data = np.array([ecog_data])
    print(ecog_data.shape )

    csp = CSP(n_components=4, reg=None, log=None, norm_trace=False)
    csp.fit(ecog_data, labels)

    print(ecog_data.shape)
    print(csp.patterns_.shape)

    xx  =csp.transform(ecog_data)
    print(xx)



def ttest_cpa():
    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.pipeline import Pipeline
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import ShuffleSplit, cross_val_score

    from mne import Epochs, pick_types, events_from_annotations
    from mne.channels import make_standard_montage
    from mne.io import concatenate_raws, read_raw_edf
    from mne.datasets import eegbci
    from mne.decoding import CSP

    print(__doc__)

    # #############################################################################
    # # Set parameters and read data

    # avoid classification of evoked responses by using epochs that start 1s after
    # cue onset.
    tmin, tmax = -1., 4.
    event_id = dict(hands=2, feet=3)
    subject = 1
    runs = [6, 10, 14]  # motor imagery: hands vs feet

    raw_fnames = eegbci.load_data(subject, runs)
    raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
    eegbci.standardize(raw)  # set channel names
    montage = make_standard_montage('standard_1005')
    raw.set_montage(montage)

    # Apply band-pass filter
    raw.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')

    events, _ = events_from_annotations(raw, event_id=dict(T1=2, T2=3))

    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')

    # Read epochs (train will be done only between 1 and 2s)
    # Testing will be done with a running classifier
    epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                    baseline=None, preload=True)
    epochs_train = epochs.copy().crop(tmin=1., tmax=2.)
    labels = epochs.events[:, -1] - 2
    epochs_data = epochs.get_data()
    epochs_data_train = epochs_train.get_data()
    cv = ShuffleSplit(10, test_size=0.2, random_state=42)
    cv_split = cv.split(epochs_data_train)

    # Assemble a classifier
    lda = LinearDiscriminantAnalysis()
    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

    # Use scikit-learn Pipeline with cross_val_score function
    clf = Pipeline([('CSP', csp), ('LDA', lda)])
    scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=None)

    clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=None)

    clf.fit(epochs_data_train, labels)
    importtances = clf.feature_importances_

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import xgboost

    clf = xgboost.XGBClassifier()

    a = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    # 求四分位数
    q1 = np.percentile(importtances, 25)
    median = np.median(importtances)


    # Printing the results
    class_balance = np.mean(labels == labels[0])
    class_balance = max(class_balance, 1. - class_balance)
    print("Classification accuracy: %f / Chance level: %f" % (np.mean(scores),
                                                              class_balance))

    # plot CSP patterns estimated on full data for visualization
    csp.fit_transform(epochs_data, labels)

    csp_data = csp.transform(epochs_data)
    print(csp_data.shape)

    csp.plot_patterns(epochs.info, ch_type='eeg', units='Patterns (AU)', size=1.5)

    print(epochs_data.shape)
    print(labels.shape)

if __name__ == '__main__':
    # main()
    ttest_cpa()
