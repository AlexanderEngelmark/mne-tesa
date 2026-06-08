import numpy as np
import plotly.graph_objects as go


def tesa_plotly_gfp(evoked, times, xlim):
    """Plot Global Field Power (GFP) using Plotly.

    This function generates an interactive plot of the Global Field Power (GFP)
    for evoked data using Plotly.

    Args:
        evoked: The evoked data instance (mne.Evoked).
        times: The time array corresponding to the evoked data.
        xlim: The time range to plot (list of start and end times in seconds).

    Example:
        >>> plot_gfp(evoked, times, xlim=[-0.1, 0.3])
    """
    # mne.Evoked, 1DArray, list

    idx_s, idx_e = evoked.time_as_index(xlim)
    data = evoked.get_data()  # Get the evoked data as a numpy array
    times = times[idx_s:idx_e] * 1000  # Get the 1-dimensional time array
    gfp_data = (
        np.std(data, axis=0, ddof=0) * 1e6
        # Just calculating the standard deviation across the time (x) axis,
        # rescaling to
        # μV from V and specifying the population std
    )
    data_to_plot = gfp_data[idx_s:idx_e]
    fig = go.Figure()
    fig.update_layout(
        title=dict(text="Global Field Power (GFP) before and after stimulation"),
        xaxis=dict(title=dict(text="Time (ms)")),
        yaxis=dict(title=dict(text='GFP (µV)"')),
    )
    fig.add_trace(
        go.Scatter(
            x=times,
            y=data_to_plot,
            mode="lines",
            name="lines",
            hovertemplate="Time: %{x:.1f} ms<br>GFP: %{y:.3f} µV<extra></extra>",
        )
    )
    fig.add_vline(
        x=0,
        line_width=0.5,
        line_dash="dash",
        line_color="black",
        annotation_text="stimulation",
    )
    fig.show()


def tesa_plotly_evoked(evoked, times, xlim=[-0.4, 0.4], width=1000, height=600):
    """Plot evoked responses using Plotly.

    This function generates an interactive plot of evoked responses for all
    channels using Plotly.

    Args:
        evoked: The evoked data instance (mne.Evoked).
        times: The time array corresponding to the evoked data.
        xlim (list, optional): The time range to plot (in seconds).
                             Defaults to [-0.4, 0.4].
        width (int, optional): The width of the plot in pixels.
                             Defaults to 1000.
        height (int, optional): The height of the plot in pixels.
                              Defaults to 600.

    Example:
        >>> plot_evoked(evoked, times, xlim=[-0.1, 0.3], width=800, height=500)
    """
    idx_s, idx_e = evoked.time_as_index(xlim)
    data = evoked.get_data()  # Get the evoked data as a numpy array
    times = times[idx_s:idx_e] * 1000  # Get the 1-dimensional time array

    fig = go.Figure()
    fig.update_layout(
        title=dict(text="Evoked response before and after stimulation"),
        xaxis=dict(title=dict(text="Time (ms)")),
        yaxis=dict(title=dict(text='Amplitude (µV)"')),
        width=width,
        height=height,
    )

    for channel, channel_name in zip(data, evoked.ch_names):
        data_to_plot = channel[idx_s:idx_e]
        fig.add_trace(
            go.Scatter(
                x=times,
                y=data_to_plot,
                mode="lines",
                name=channel_name,
                hovertemplate="Time: %{x:.1f} ms<br>"
                + channel_name
                + ": %{y:.3f} µV<extra></extra>",
            )
        )
    fig.add_vline(
        x=0,
        line_width=0.5,
        line_dash="dash",
        line_color="black",
        annotation_text="stimulation",
    )
    fig.show()
