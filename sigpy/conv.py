# -*- coding: utf-8 -*-
"""Convolution functions with multi-dimension, and multi-channel support.

Changes vs original sigpy conv.py:
  - cuDNN support removed entirely; the cudnn dependency and config import
    are gone, along with all _cuda private functions and _complex helper.
  - All three public functions now unconditionally call the CPU (_) variants.
  - The CPU (_) variants dispatch to cupyx.scipy.signal on GPU arrays so
    that GPU inputs remain on-device (no asnumpy round-trip). CPU inputs
    continue to use scipy.signal as before.
  - Everything else is identical to the original.
"""
import numpy as np
import scipy.signal as signal

from sigpy import backend, util

__all__ = ["convolve", "convolve_data_adjoint", "convolve_filter_adjoint"]


# ---------------------------------------------------------------------------
# Signal-dispatch helper
# ---------------------------------------------------------------------------

def _get_signal_module(xp):
    """Return cupyx.scipy.signal for CuPy arrays, scipy.signal otherwise."""
    if xp is np:
        return signal
    try:
        import cupyx.scipy.signal as cupy_signal
        return cupy_signal
    except ImportError:
        raise RuntimeError(
            "cupyx.scipy.signal is required for GPU convolution without cuDNN. "
            "Install CuPy with: pip install cupy-cuda12x"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def convolve(data, filt, mode="full", strides=None, multi_channel=False):
    r"""Convolution that supports multi-dimensional and multi-channel inputs.

    This function follows the signal processing definition of convolution.

    Args:
        data (array): data array of shape:
            :math:`[..., m_1, ..., m_D]` if multi_channel is False,
            :math:`[..., c_i, m_1, ..., m_D]` otherwise.
        filt (array): filter array of shape:
            :math:`[n_1, ..., n_D]` if multi_channel is False
            :math:`[c_o, c_i, n_1, ..., n_D]` otherwise.
        mode (str): {'full', 'valid'}.
        strides (None or tuple of ints): convolution strides of length D.
        multi_channel (bool): specify if input/output has multiple channels.

    Returns:
        array: output array of shape:
            :math:`[..., p_1, ..., p_D]` if multi_channel is False,
            :math:`[..., c_o, p_1, ..., p_D]` otherwise.
    """
    return _convolve(data, filt, mode=mode, strides=strides, multi_channel=multi_channel)


def convolve_data_adjoint(
    output, filt, data_shape, mode="full", strides=None, multi_channel=False
):
    """Adjoint convolution operation with respect to data.

    Args:
        output (array): output array of shape
            :math:`[..., p_1, ..., p_D]` if multi_channel is False,
            :math:`[..., c_o, p_1, ..., p_D]` otherwise.
        filt (array): filter array of shape
            :math:`[n_1, ..., n_D]` if multi_channel is False
            :math:`[c_o, c_i, n_1, ..., n_D]` otherwise.
        mode (str): {'full', 'valid'}.
        strides (None or tuple of ints): convolution strides of length D.
        multi_channel (bool): specify if data/output has multiple channels.

    Returns:
        array: data array of shape
            :math:`[..., m_1, ..., m_D]` if multi_channel is False,
            :math:`[..., c_i, m_1, ..., m_D]` otherwise.
    """
    data_shape = tuple(data_shape)
    return _convolve_data_adjoint(
        output, filt, data_shape, mode=mode, strides=strides, multi_channel=multi_channel
    )


def convolve_filter_adjoint(
    output, data, filt_shape, mode="full", strides=None, multi_channel=False
):
    """Adjoint convolution operation with respect to filter.

    Args:
        output (array): output array of shape:
            :math:`[..., p_1, ..., p_D]` if multi_channel is False,
            :math:`[..., c_o, p_1, ..., p_D]` otherwise.
        data (array): data array of shape:
            :math:`[..., m_1, ..., m_D]` if multi_channel is False,
            :math:`[..., c_i, m_1, ..., m_D]` otherwise.
        mode (str): {'full', 'valid'}.
        strides (None or tuple of ints): convolution strides of length D.
        multi_channel (bool): specify if input/output has multiple channels.

    Returns:
        array: filter array of shape:
            :math:`[n_1, ..., n_D]` if multi_channel is False
            :math:`[c_o, c_i, n_1, ..., n_D]` otherwise.
    """
    filt_shape = tuple(filt_shape)
    return _convolve_filter_adjoint(
        output, data, filt_shape, mode=mode, strides=strides, multi_channel=multi_channel
    )


# ---------------------------------------------------------------------------
# Parameter helper — unchanged from original
# ---------------------------------------------------------------------------

def _get_convolve_params(data_shape, filt_shape, mode, strides, multi_channel):
    D = len(filt_shape) - 2 * multi_channel
    m = tuple(data_shape[-D:])
    n = tuple(filt_shape[-D:])
    b = tuple(data_shape[: -D - multi_channel])
    B = util.prod(b)

    if multi_channel:
        if filt_shape[-D - 1] != data_shape[-D - 1]:
            raise ValueError(
                "Data channel mismatch, "
                "got {} from data and {} from filt.".format(
                    data_shape[-D - 1], filt_shape[-D - 1]
                )
            )
        c_i = filt_shape[-D - 1]
        c_o = filt_shape[-D - 2]
    else:
        c_i = 1
        c_o = 1

    if strides is None:
        s = (1,) * D
    else:
        if len(strides) != D:
            raise ValueError("Strides must have length {}.".format(D))
        s = tuple(strides)

    if mode == "full":
        p = tuple(
            (m_d + n_d - 1 + s_d - 1) // s_d
            for m_d, n_d, s_d in zip(m, n, s)
        )
    elif mode == "valid":
        if any(m_d >= n_d for m_d, n_d in zip(m, n)) and any(
            m_d < n_d for m_d, n_d in zip(m, n)
        ):
            raise ValueError(
                "In valid mode, either data or filter must be "
                "at least as large as the other in every axis."
            )
        p = tuple(
            (m_d - n_d + 1 + s_d - 1) // s_d
            for m_d, n_d, s_d in zip(m, n, s)
        )
    else:
        raise ValueError("Invalid mode, got {}".format(mode))

    return D, b, B, m, n, s, c_i, c_o, p


# ---------------------------------------------------------------------------
# Implementations — CPU and GPU unified via _get_signal_module dispatch
# ---------------------------------------------------------------------------

def _convolve(data, filt, mode="full", strides=None, multi_channel=False):
    xp = backend.get_array_module(data)
    sig = _get_signal_module(xp)

    D, b, B, m, n, s, c_i, c_o, p = _get_convolve_params(
        data.shape, filt.shape, mode, strides, multi_channel
    )

    data = data.reshape((B, c_i) + m)
    filt = filt.reshape((c_o, c_i) + n)
    output = xp.zeros((B, c_o) + p, dtype=data.dtype)
    slc = tuple(slice(None, None, s_d) for s_d in s)

    for k in range(B):
        for j in range(c_o):
            for i in range(c_i):
                output[k, j] += sig.convolve(
                    data[k, i], filt[j, i], mode=mode
                )[slc]

    if multi_channel:
        output = output.reshape(b + (c_o,) + p)
    else:
        output = output.reshape(b + p)

    return output


def _convolve_data_adjoint(
    output, filt, data_shape, mode="full", strides=None, multi_channel=False
):
    xp = backend.get_array_module(output)
    sig = _get_signal_module(xp)

    D, b, B, m, n, s, c_i, c_o, p = _get_convolve_params(
        data_shape, filt.shape, mode, strides, multi_channel
    )

    output = output.reshape((B, c_o) + p)
    filt = filt.reshape((c_o, c_i) + n)
    data = xp.zeros((B, c_i) + m, dtype=output.dtype)
    slc = tuple(slice(None, None, s_d) for s_d in s)

    if mode == "full":
        output_kj = xp.zeros(
            [m_d + n_d - 1 for m_d, n_d in zip(m, n)], dtype=output.dtype
        )
        adjoint_mode = "valid"
    elif mode == "valid":
        output_kj = xp.zeros(
            [max(m_d, n_d) - min(m_d, n_d) + 1 for m_d, n_d in zip(m, n)],
            dtype=output.dtype,
        )
        if all(m_d >= n_d for m_d, n_d in zip(m, n)):
            adjoint_mode = "full"
        else:
            adjoint_mode = "valid"

    for k in range(B):
        for j in range(c_o):
            for i in range(c_i):
                output_kj[slc] = output[k, j]
                data[k, i] += sig.correlate(
                    output_kj, filt[j, i], mode=adjoint_mode
                )

    data = data.reshape(data_shape)
    return data


def _convolve_filter_adjoint(
    output, data, filt_shape, mode="full", strides=None, multi_channel=False
):
    xp = backend.get_array_module(data)
    sig = _get_signal_module(xp)

    D, b, B, m, n, s, c_i, c_o, p = _get_convolve_params(
        data.shape, filt_shape, mode, strides, multi_channel
    )

    data = data.reshape((B, c_i) + m)
    output = output.reshape((B, c_o) + p)
    slc = tuple(slice(None, None, s_d) for s_d in s)

    if mode == "full":
        output_kj = xp.zeros(
            [m_d + n_d - 1 for m_d, n_d in zip(m, n)], dtype=output.dtype
        )
        adjoint_mode = "valid"
    elif mode == "valid":
        output_kj = xp.zeros(
            [max(m_d, n_d) - min(m_d, n_d) + 1 for m_d, n_d in zip(m, n)],
            dtype=output.dtype,
        )
        if all(m_d >= n_d for m_d, n_d in zip(m, n)):
            adjoint_mode = "valid"
        else:
            adjoint_mode = "full"

    filt = xp.zeros((c_o, c_i) + n, dtype=output.dtype)
    for k in range(B):
        for j in range(c_o):
            for i in range(c_i):
                output_kj[slc] = output[k, j]
                filt[j, i] += sig.correlate(
                    output_kj, data[k, i], mode=adjoint_mode
                )

    filt = filt.reshape(filt_shape)
    return filt