import numpy as np
import mne
import pandas as pd
import matplotlib.pyplot as plt

from mne.preprocessing import ICA
from scipy.stats import zscore

from typing import Optional


def notch_filter_epochs(epochs):
    from mne.filter import construct_iir_filter, filter_data

    iir_params = construct_iir_filter(
        iir_params=dict(order=4, ftype="butter", output="sos"),
        f_pass=[48, 52],
        f_stop=None,
        sfreq=epochs.info["sfreq"],
        btype="bandstop",
        phase="zero",
    )

    def apply_bandstop(x):
        return filter_data(
            x,
            sfreq=epochs.info["sfreq"],
            l_freq=None,
            h_freq=None,
            method="iir",
            iir_params=iir_params,
            phase="zero",
        )

    epochs.apply_function(apply_bandstop, picks="all", channel_wise=True)

    return epochs


def tesa_interp_data_cubic_spline(
    inst,
    inst_type: str,
    tmin: Optional[int],
    tmax: Optional[int],
    events: Optional[np.ndarray],
):
    from scipy.interpolate import CubicSpline

    if inst_type == "raw":
        inst.load_data()

        def remove_tms(y):
            s_start = int(np.ceil(inst.info["sfreq"] * tmin))
            s_end = int(np.ceil(inst.info["sfreq"] * tmax))
            for event_idx in events[1:, 0]:
                first_samp = int(event_idx) - inst.first_samp + s_start
                last_samp = int(event_idx) - inst.first_samp + s_end
                x = np.array([first_samp, last_samp])
                f = CubicSpline(x, y[[first_samp, last_samp]])
                xnew = np.arange(first_samp, last_samp)
                interp_data = f(xnew)
                y[first_samp:last_samp] = interp_data
            return y

        inst.apply_function(remove_tms, picks="all", verbose=False)

    elif inst_type == "epochs":
        s_start = int(np.ceil(inst.info["sfreq"] * tmin))
        s_end = int(np.ceil(inst.info["sfreq"] * tmax))
        e_start = int(np.ceil(inst.info["sfreq"] * inst.tmin))
        first_samp = s_start - e_start
        last_samp = s_end - e_start

        def remove_tms_epoch(y):
            x = np.array([first_samp, last_samp])
            f = CubicSpline(x, y[[first_samp, last_samp]])
            xnew = np.arange(first_samp, last_samp)
            interp_data = f(xnew)
            y[first_samp:last_samp] = interp_data
            return y

        inst.apply_function(remove_tms_epoch, picks="all", verbose=False)

    return inst


def tesa_replace_constant_amplitude(
    inst,
    inst_type: str,
    tmin: Optional[int],
    tmax: Optional[int],
    events: Optional[np.ndarray],
):
    if inst_type == "raw":
        inst.load_data()

        def remove_tms(y):
            s_start = int(np.ceil(inst.info["sfreq"] * tmin))
            s_end = int(np.ceil(inst.info["sfreq"] * tmax))
            for event_idx in events[1:, 0]:
                first_samp = int(event_idx) - inst.first_samp + s_start
                last_samp = int(event_idx) - inst.first_samp + s_end
                y[first_samp:last_samp] = 0
            return y

        inst.apply_function(remove_tms, picks="all", verbose=False)

    elif inst_type == "epochs":
        s_start = int(np.ceil(inst.info["sfreq"] * tmin))
        s_end = int(np.ceil(inst.info["sfreq"] * tmax))
        e_start = int(np.ceil(inst.info["sfreq"] * inst.tmin))
        first_samp = s_start - e_start
        last_samp = s_end - e_start

        def remove_tms_epoch(y):
            y[first_samp:last_samp] = 0
            return y

        inst.apply_function(remove_tms_epoch, picks="all", verbose=False)

    return inst


def find_pulse(raw, sfreq, thresh=5, plot=False):
    # Might be useful to exclude some really bad channels
    # before if it's impossible to catch the pulse?

    thresh *= 100
    data = raw.get_data()

    from scipy.signal import butter, filtfilt  # Filter parameters

    cutoff_freq = 2  # Cutoff frequency in Hz
    fs = sfreq  # Sampling rate in Hz
    order = 4
    nyq = 0.5 * fs
    normal_cutoff = cutoff_freq / nyq
    b, a = butter(order, normal_cutoff, btype="highpass")
    filtered_data = filtfilt(b, a, data)
    times = raw.times * 1000  # Get the 1-dimensional time array
    # Just getting the GFP as in done in MNE
    gfp_data = np.std(filtered_data, axis=0, ddof=0) * 1e6

    if plot:
        plt.figure(figsize=(20, 4))
        plt.plot(times, gfp_data)

    # Just an arbitrary threshold for the GFP which should catch the TMS-pulses
    # Should just be possible to tweak that and the filter
    # so that one can catch all the pulses
    over_thres = gfp_data > thresh

    pulses = np.where(over_thres)

    Annotations = mne.Annotations(onset=0, duration=1e-6, description="START")
    for n in range(len(pulses[0])):
        idx = pulses[0][n]
        onset = idx / raw.info["sfreq"]
        Annotations.append(onset=onset, duration=1e-6, description="TMS")
    raw.set_annotations(Annotations)
    raw.annotations.delete(0)

    events_from_annot, event_dict = mne.events_from_annotations(raw)
    print(event_dict)
    print(f"Number of events: {len(events_from_annot)}")

    return events_from_annot, event_dict


class ICA_TESA(ICA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def find_bads_lateral_eye_movement(self, inst, ch_names=["F7", "F8"], threshold=2):
        #   This type of artifact is detected by comparing the z scores
        #   (calculated on the component topography weights)
        #   of two electrodes on either side
        #   of the forehead (e.g. 'F7','F8'). The z score must be positive for
        #   one and
        #   negative for the other. A threhold is set by the user for detection
        #   (e.g. 2 means the z score in one electrode must be greater than
        #   2 and less than
        #   -2 in the other electrode). Components are stored under eyes.

        eye_movement_components = []
        eye_movement_scores = []

        components = self.get_components()
        # components : array, shape (n_channels, n_components)

        move_elec_1_idx = self.ch_names.index(ch_names[0])
        move_elec_2_idx = self.ch_names.index(ch_names[1])

        # Transpose to enumerate component topography matrices and not channels
        for n, component in enumerate(components.T):
            # Default for the zscore is axis = 0
            zscores = zscore(component)

            move_elec_1_score = zscores[move_elec_1_idx]
            move_elec_2_score = zscores[move_elec_2_idx]

            if (move_elec_1_score > threshold and move_elec_2_score < -threshold) or (
                move_elec_2_score > threshold and move_elec_1_score < -threshold
            ):
                print(f"Lateral eye movement detected in component {n}")

                eye_movement_components.append(n)
                eye_movement_scores.append(
                    (move_elec_1_score, move_elec_2_score))
                # else:
                # print(f"Eye movement not detected for component {n}")

        return eye_movement_components, eye_movement_scores

    def find_bads_tms_muscle(self, inst, threshold=8, muscle_window=[0.011, 0.030]):
        # TMS-evoked muscle activity
        # This type of artifact is detected by comparing the mean absolute
        # amplitude of the component time course within a target window
        # ('tmsMuscleWin')
        # and the mean absolute amplitude across the entire component time
        # course A threshold is
        # set by the user for detection (e.g. 8 means the mean absolute
        # amplitude in the target window
        # is 8 times larger than the mean absolute amplitude across the entire
        # time course).
        # Components are stored under tmsMuscle.

        tms_muscle_components = []

        muscle_window_idx = inst.time_as_index(muscle_window)

        # Sources: Raw, Epochs or Evoked
        # inst.get_sources() return a MNE.Epochs object
        # when inst are epochs
        sources = self.get_sources(inst)

        # average across epochs
        sources_evoked = sources.average(sources.ch_names)

        # n_components, n_times
        sources_data = sources_evoked.get_data()

        for n, component in enumerate(sources_data):
            mean_abs_amp_win = np.mean(
                np.abs(component[muscle_window_idx[0]: muscle_window_idx[1]])
            )
            component_mean_abs_amp = (np.mean(np.abs(component))) * threshold
            if mean_abs_amp_win > component_mean_abs_amp:
                print(
                    f"Component {
                        n
                    } marked as TMS muscle\n Mean absolute amplitude in muscle window {
                        mean_abs_amp_win
                    } mean absolute amplitude in component {
                        component_mean_abs_amp / threshold
                    }"
                )
                tms_muscle_components.append(n)

        return tms_muscle_components

    def find_bads_electrode_noise(self, inst, threshold=4):
        #   Electrode noise
        #   This type of artifact is detected by comparing z scores in
        #   individual
        #   electrodes (calculated on the component topography weights). A
        #   threhold is
        #   set by the user for detection (e.g. 4 means one or more electrodes
        #   has
        #   an absolute z score of at least 4).

        elec_noise_components = []

        components = self.get_components()

        # Transpose to enumerate component topography
        # matrices and not channels
        for n, component in enumerate(components.T):
            # Default for the zscore is axis = 0
            zscore_comp = zscore(component)

            if np.any(np.abs(zscore_comp) > threshold):
                # print(np.max(np.abs(zscore_comp)))
                elec_noise_components.append(n)

        return elec_noise_components


def mne_tesa_class_comp(
    epochs,
    ica: ICA_TESA,
    tmsMuscle=True,
    tmsMuscleThresh=8,  # 8 was the default in TESA
    tmsMuscleWin=[0.011, 0.030],
    blink=True,
    blinkThresh=2.5,
    blinkElecs=["Fp1", "Fp2"],
    lat_eye_move=True,
    lat_eye_moveThresh=2.0,
    lat_eye_move_elecs=["F7", "F8"],
    persistant_muscle=True,
    persistant_muscle_thresh=0.5,  # original in TESA was -0.31,
    muscleFreqIn=[7, 70],  # original in TESA was [7, 70]
    # muscleFreqEx=[48, 52], ## no exclusion made in MNE implementation
    electrode_noise=True,
    elec_noise_thresh=4,
):
    classified_comps = {}

    if lat_eye_move:
        lat_eye_move_comps, lat_eye_movement_scores = (
            ica.find_bads_lateral_eye_movement(
                epochs, lat_eye_move_elecs, lat_eye_moveThresh
            )
        )
        classified_comps["lat_eye_move"] = lat_eye_move_comps

    if tmsMuscle:
        tms_muscle_comps = ica.find_bads_tms_muscle(
            epochs, tmsMuscleThresh, tmsMuscleWin
        )
        classified_comps["tmsMuscle"] = tms_muscle_comps

    if persistant_muscle:
        persistant_muscle_comps, persistant_muscle_scores = ica.find_bads_muscle(
            epochs,
            threshold=0.5,
            start=None,
            stop=None,
            l_freq=muscleFreqIn[0],
            h_freq=muscleFreqIn[1],
            sphere="eeglab",
            verbose=None,
        )

        classified_comps["persistant_muscle"] = persistant_muscle_comps

    if blink:
        maybe_double_eog_components = set()

        for channel in blinkElecs:
            eog_idx, scores = ica.find_bads_eog(
                epochs,
                ch_name=channel,
                threshold=3.0,
                start=None,
                stop=None,
                l_freq=1,
                h_freq=10,
                reject_by_annotation=True,
                measure="zscore",
                verbose=None,
            )
            maybe_double_eog_components.update(eog_idx)

        eog_comps = list(maybe_double_eog_components)

        classified_comps["blink"] = eog_comps

    if electrode_noise:
        elec_noise_components = ica.find_bads_electrode_noise(
            epochs, elec_noise_thresh)
        classified_comps["elecNoise"] = elec_noise_components

    rows = []
    for category, numbers in classified_comps.items():
        for number in numbers:
            rows.append((number, category))

    df = pd.DataFrame(rows, columns=["number", "category"])

    grouped = df.groupby("number")["category"].apply(list).reset_index()

    return classified_comps, grouped


def tesa_ica_select(epochs, ICA, tesa_comp_class_result=None):
    if tesa_comp_class_result is None:
        tesa_comp_class_result = {}

    global ica
    ica = ICA
    global original_epochs
    original_epochs = epochs.copy()
    global to_remove
    to_remove = []

    # epochs.plot();

    def remove_comp(n):
        global to_remove, original_epochs, ica
        to_remove.append(n)
        temp_epochs = original_epochs.copy()
        ica.apply(temp_epochs, exclude=to_remove)

        # temp_epochs.average().plot();
        ica.plot_overlay(original_epochs.average(), exclude=to_remove)

    import ipywidgets as widgets
    from IPython.display import display

    number_of_components = list(range(ica.n_components))
    ncs = [str(n) for n in number_of_components]

    output = widgets.Output()
    w = widgets.Dropdown(
        options=ncs,
        value="0",
        description="Component index:",
        disabled=False,
    )

    def on_change(change):
        if change["type"] == "change" and change["name"] == "value":
            selected_comp = int(change["new"])
            output.clear_output()
            with output:
                ica.plot_properties(original_epochs, picks=selected_comp)
                labels = []
                for key, value in tesa_comp_class_result.items():
                    if selected_comp in [int(item) for item in value]:
                        labels.append(key)
                if labels == []:
                    print("\n\n Component was not flagged \n\n")
                else:
                    print(
                        f"\n************\nComponent is flagged as: {
                            labels
                        }\n\n************\n"
                    )

    w.observe(on_change)

    b = widgets.Button(
        description="Exclude",
        disabled=False,
        button_style="",
        tooltip="Exclude ica component and plot overlay",
        icon="check",
    )

    def on_button_clicked(b):
        output.clear_output()
        with output:
            remove_comp(int(w.value))

    b.on_click(on_button_clicked)
    display(w, b, output)

    return to_remove
