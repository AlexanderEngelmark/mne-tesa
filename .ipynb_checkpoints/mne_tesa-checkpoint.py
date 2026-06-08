import numpy as np
import mne
import pandas as pd
import matplotlib.pyplot as plt

from mne.preprocessing import ICA
from scipy.stats import zscore

from scipy.interpolate import CubicSpline

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


def tesa_interp_cubic_spline(
    inst,
    inst_type: str,
    tmin: float,
    tmax: float,
    interp_win: list = [0.001, 0.001],  # seconds before and after the gap
    events: np.ndarray = None,
):
    """Interpolate TMS pulse artifacts using cubic splines.

    For each TMS event, the data in the interval [tmin, tmax] (relative to the event)
    is replaced by a cubic spline fitted to `n_pre` samples before and `n_post` samples
    after the interval.

    Parameters
    ----------
    inst : mne.io.Raw or mne.Epochs
        The data container.
    inst_type : {'raw', 'epochs'}
        Type of the instance.
    tmin, tmax : float
        Start and end of the interpolation window (in seconds), relative to each event.
    events : np.ndarray, optional
        Required for raw data. Must be an array of event samples (e.g., from
        mne.find_events). For epochs, this is ignored.
    n_pre, n_post : int
        Number of samples to take before and after the gap for fitting the spline.
        Default is 5, which is typical for short gaps.

    Returns
    -------
    inst : mne.io.Raw or mne.Epochs
        The modified instance.
    """
    sfreq = inst.info["sfreq"]

    s_start = int(np.round(sfreq * tmin))
    s_end = int(np.round(sfreq * tmax))

    n_pre = int(np.round(sfreq * interp_win[0]))
    n_post = int(np.round(sfreq * interp_win[1]))

    if inst_type == "raw":
        if events is None:
            raise ValueError("For raw data, you must provide the 'events' array.")
        event_samples = events[:, 0]
        data = inst.get_data()
        n_channels, n_samples = data.shape
        times = inst.times  # absolute times

        for ev in event_samples:
            gap_start = ev + s_start
            gap_end = ev + s_end
            if gap_start < 0 or gap_end >= n_samples:
                continue

            start_wide = max(0, gap_start - n_pre)
            end_wide = min(n_samples, gap_end + n_post)

            times_wide = times[start_wide:end_wide]
            t_shift = times_wide - times_wide[0]

            gap_local_start = gap_start - start_wide
            gap_local_end = gap_end - start_wide

            for ch in range(n_channels):
                y_wide = data[ch, start_wide:end_wide]

                known_mask = np.ones(len(y_wide), dtype=bool)
                known_mask[gap_local_start:gap_local_end] = False
                x_known = t_shift[known_mask]
                y_known = y_wide[known_mask]

                if len(x_known) >= 4:
                    # Fit cubic polynomial (least squares if >4 points)
                    coeffs = np.polyfit(x_known, y_known, 3)
                    x_gap = t_shift[gap_local_start:gap_local_end]
                    y_gap = np.polyval(coeffs, x_gap)
                    data[ch, gap_start:gap_end] = y_gap

        inst._data = data
        print(f"Interpolated data between {tmin} and {tmax} using np.polyfit")

    elif inst_type == "epochs":
        epoch_start = inst.times[0]
        first_samp = int(np.round(sfreq * tmin)) - int(np.round(sfreq * epoch_start))
        last_samp = int(np.round(sfreq * tmax)) - int(np.round(sfreq * epoch_start))

        gap_idx = np.arange(first_samp, last_samp)

        data = inst.get_data()  # shape (n_epochs, n_channels, n_times)
        n_epochs, n_channels, n_times = data.shape
        times = inst.times  # epoch‑relative times

        for ep in range(n_epochs):
            for ch in range(n_channels):
                start_wide = max(0, first_samp - n_pre)
                end_wide = min(n_times, last_samp + n_post)

                if end_wide - start_wide < 4:
                    continue

                times_wide = times[start_wide:end_wide]
                t_shift = times_wide - times_wide[0]

                gap_local_start = first_samp - start_wide
                gap_local_end = last_samp - start_wide

                y_wide = data[ep, ch, start_wide:end_wide]
                known_mask = np.ones(len(y_wide), dtype=bool)
                known_mask[gap_local_start:gap_local_end] = False
                x_known = t_shift[known_mask]
                y_known = y_wide[known_mask]

                if len(x_known) >= 4:
                    coeffs = np.polyfit(x_known, y_known, 3)
                    x_gap = t_shift[gap_local_start:gap_local_end]
                    y_gap = np.polyval(coeffs, x_gap)
                    data[ep, ch, first_samp:last_samp] = y_gap

        inst._data = data
        print(f"Interpolated data between {tmin} and {tmax} using np.polyfit")

    else:
        raise ValueError("inst_type must be 'raw' or 'epochs'")

    return inst


def tesa_replace_constant_amplitude(
    inst,
    inst_type: str,
    tmin: Optional[int],
    tmax: Optional[int],
    events: Optional[np.ndarray],
):
    """Replace data with constant amplitude (zero).

    This function replaces data around specified time points with zeros
    to allow for filtering, ICA, and downsampling. It supports two types
    of input data: "mne.Raw" and "mne.Epochs".

    Args:
        inst: The instance containing the data to be modified.
        inst_type (str): The type of data, either "raw" or "epochs".
        tmin (Optional[int]): The start time for replacement.
        tmax (Optional[int]): The end time for replacement.
        events (Optional[np.ndarray]): The events to use for replacement
        in "raw" mode.

    Returns:
        The instance with modified data.

    Raises:
        ValueError: If `inst_type` is not "raw" or "epochs".

    Example:
        >>> tesa_replace_constant_amplitude(epochs, "epochs", -2, 15, events)
    """
    sfreq = inst.info["sfreq"]
    s_start = int(np.round(sfreq * tmin))
    s_end = int(np.round(sfreq * tmax))

    if inst_type == "raw":
        if events is None:
            raise ValueError("For raw data, events must be provided.")
        inst.load_data()
        event_samples = events[:, 0]  # absolute sample indices
        data = inst.get_data()
        n_channels, n_samples = data.shape
        for ev in event_samples:
            first_samp = ev + s_start
            last_samp = ev + s_end
            if 0 <= first_samp < last_samp <= n_samples:
                data[:, first_samp:last_samp] = 0
        inst._data = data

    elif inst_type == "epochs":
        epoch_start = inst.times[0]
        first_samp = int(np.round(sfreq * tmin)) - int(np.round(sfreq * epoch_start))
        last_samp = int(np.round(sfreq * tmax)) - int(np.round(sfreq * epoch_start))
        data = inst.get_data()
        data[:, :, first_samp:last_samp] = 0
        inst._data = data

    else:
        raise ValueError("inst_type must be 'raw' or 'epochs'")
    return inst


def find_pulse(raw, sfreq, thresh=5, plot=False):
    """Detect TMS pulses in raw data using GFP thresholding.

    This function identifies TMS pulses in raw EEG data by applying a high-pass
    filter
    and using Global Field Power (GFP) thresholding to detect pulse events.

    Args:
        raw: The raw EEG data instance (mne.io.Raw).
        sfreq (float): The sampling frequency of the data in Hz.
        thresh (float, optional): The threshold for pulse detection
        (in microvolts).
                                 Defaults to 5.
        plot (bool, optional): Whether to plot the GFP data for visualization.
                              Defaults to False.

    Returns:
        tuple: A tuple containing:
            - events_from_annot (np.ndarray): Array of detected events.
            - event_dict (dict): Dictionary mapping event descriptions to event
            codes.

    Example:
        >>> events, event_dict = find_pulse(raw, sfreq, thresh=5, plot=True)
    """
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
    times = raw.times * 1000
    # Just getting the GFP as in done in MNE
    gfp_data = np.std(filtered_data, axis=0, ddof=0) * 1e6

    if plot:
        plt.figure(figsize=(20, 4))
        plt.plot(times, gfp_data)

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
        """Detect lateral eye movement artefacts in ICA components.

        This function identifies lateral eye movement artefacts by comparing
        z-scores of specified frontal electrodes (default: 'F7' and 'F8')
        in the component topographies. The z-score must be positive for one
        electrode and negative for the other, with both exceeding the
        specified threshold.

        Args:
            inst: The data instance (Raw or Epochs).
            ch_names (list, optional): List of two channel names to compare.
                                    Defaults to ["F7", "F8"].
            threshold (float, optional): Threshold for z-score detection.
                                        Defaults to 2.

        Returns:
            tuple: A tuple containing:
                - list: Indices of components with lateral eye movement artefacts.
                - list: Z-score pairs for each detected component.

        Example:
            >>> eye_components, eye_scores = ica.find_bads_lateral_eye_movement(epochs, ch_names=["F7", "F8"], threshold=2)
        """

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
                eye_movement_scores.append((move_elec_1_score, move_elec_2_score))
                # else:
                # print(f"Eye movement not detected for component {n}")

        return eye_movement_components, eye_movement_scores

    def find_bads_tms_muscle(self, inst, threshold=8, muscle_window=[0.011, 0.030]):
        """Detect TMS-evoked muscle activity artefacts in ICA components.

        This function identifies muscle activity artifacts by comparing the mean
        absolute amplitude of the component time course within a target
        time window
        to the overall mean amplitude of the component time course.
        A component is
        flagged if the mean amplitude in the target window exceeds the overall mean
        amplitude by the specified threshold factor.

        Args:
            inst: The data instance (Raw or Epochs).
            threshold (float, optional): Threshold for amplitude ratio detection.
                                        Defaults to 8.
            muscle_window (list, optional): Time window to check for muscle activity.
                                        Defaults to [0.011, 0.030].

        Returns:
            list: Indices of components with TMS muscle artefacts.

        Example:
            >>> muscle_components = ica.find_bads_tms_muscle(epochs, threshold=8, muscle_window=[0.011, 0.030])
        """

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
                np.abs(component[muscle_window_idx[0] : muscle_window_idx[1]])
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
        """Detect electrode noise artifacts in ICA components.

        This function identifies components with electrode noise by enumerating the
        component topographies and checking for any channel index with an absolute z-score
        exceeding the specified threshold within these topographies.

        Args:
            inst: The data instance (Raw or Epochs).
            threshold (float, optional): Threshold for z-score detection.
                                        Defaults to 4.

        Returns:
            list: Indices of components with electrode noise artifacts.

        Example:
            >>> noise_components = ica.find_bads_electrode_noise(epochs, threshold=4)
        """

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
    """Classify ICA components using multiple artefact detection methods.

    This wrapper function applies various artifact detection methods to classify
    ICA components in epochs data using the ICA_TESA instance. It also includes
    the mne.preprocessing.ica.find_bads_muscle() as well as the mne.preprocessing.find_bads_eog()
    methods and we refer to https://mne.tools/stable/generated/mne.preprocessing.ICA.html#mne.preprocessing.ICA
    for information about these methods.

    Args:
        epochs: The epochs data instance.
        ica (ICA_TESA): The ICA_TESA instance for artifact detection.
        tmsMuscle (bool, optional): Whether to detect TMS muscle artifacts.
                                  Defaults to True.
        tmsMuscleThresh (float, optional): Threshold for TMS muscle detection.
                                         Defaults to 8.
        tmsMuscleWin (list, optional): Time window for TMS muscle detection.
                                     Defaults to [0.011, 0.030].
        blink (bool, optional): Whether to detect blink artifacts.
                              Defaults to True.
        blinkThresh (float, optional): Threshold for blink detection.
                                     Defaults to 2.5.
        blinkElecs (list, optional): Channels for blink detection.
                                   Defaults to ["Fp1", "Fp2"].
        lat_eye_move (bool, optional): Whether to detect lateral eye movements.
                                     Defaults to True.
        lat_eye_moveThresh (float, optional): Threshold for eye movement detection.
                                            Defaults to 2.0.
        lat_eye_move_elecs (list, optional): Channels for eye movement detection.
                                           Defaults to ["F7", "F8"].
        persistant_muscle (bool, optional): Whether to detect persistent muscle activity.
                                          Defaults to True.
        persistant_muscle_thresh (float, optional): Threshold for muscle detection.
                                                  Defaults to 0.5.
        muscleFreqIn (list, optional): Frequency range for muscle detection.
                                    Defaults to [7, 70].
        electrode_noise (bool, optional): Whether to detect electrode noise.
                                        Defaults to True.
        elec_noise_thresh (float, optional): Threshold for electrode noise detection.
                                           Defaults to 4.

    Returns:
        tuple: A tuple containing:
            - dict: Dictionary of classified components by artifact type.
            - DataFrame: Grouped DataFrame of component classifications.

    Example:
        >>> classified_components, dataframe = mne_tesa_class_comp(
            epochs,
            ica,
            tmsMuscle=True,
            tmsMuscleThresh=8,
            tmsMuscleWin=[0.011, 0.030],
            blink=True,
            blinkThresh=2.5,
            blinkElecs=["Fp1", "Fp2"],
            lat_eye_move=True,
            lat_eye_moveThresh=2.0,
            lat_eye_move_elecs=["F7", "F8"],
            persistant_muscle=True,
            persistant_muscle_thresh=0.5,
            muscleFreqIn=[7, 70],
            electrode_noise=True,
            elec_noise_thresh=4,
        )
    """
    classified_comps = {}
    if lat_eye_move:
        missing = [ch for ch in lat_eye_move_elecs if ch not in epochs.ch_names]
        if missing:
            warnings.warn(
                f"Lateral eye movement electrodes {missing} not found. "
                "Skipping lateral eye movement detection."
            )
        else:
            lat_eye_move_comps, lat_eye_movement_scores = ica.find_bads_lateral_eye_movement(
                epochs, lat_eye_move_elecs, lat_eye_moveThresh
            )
            classified_comps["lat_eye_move"] = lat_eye_move_comps
            
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
        missing = [ch for ch in blinkElecs if ch not in epochs.ch_names]
        if missing:
            warnings.warn(
                f"Blink electrodes {missing} not found. "
                "Skipping blink detection."
            )
        else:
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
        elec_noise_components = ica.find_bads_electrode_noise(epochs, elec_noise_thresh)
        classified_comps["elecNoise"] = elec_noise_components

    rows = []
    for category, numbers in classified_comps.items():
        for number in numbers:
            rows.append((number, category))

    df = pd.DataFrame(rows, columns=["number", "category"])

    grouped = df.groupby("number")["category"].apply(list).reset_index()

    return classified_comps, grouped


def tesa_ica_select(epochs, ICA, tesa_comp_class_result=None):
    """Interactive ICA component selection and visualization for Jupyter notebooks.

    This function creates an interactive widget for visualizing and selecting
    ICA components to exclude from epochs data. It displays component properties
    and classification labels from the tesa_comp_class_result.

    Note: This function requires ipywidgets to be installed.

    Args:
        epochs: The epochs data instance.
        ICA: The ICA instance for artifact detection.
        tesa_comp_class_result (dict, optional): Dictionary of classified components
                                              from mne_tesa_class_comp.
                                              Defaults to None.

    Returns:
        list: List of component indices to exclude.

    Example:
        >>> to_remove = tesa_ica_select(epochs, ica, tesa_comp_class_result)
    """

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
