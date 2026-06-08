# -*- coding: utf-8 -*-
"""Convolution functions with multi-dimension, and multi-channel support.

Changes vs original sigpy conv.py:
  - cuDNN support removed entirely; the cudnn dependency and config import
    are gone, along with all _cuda private functions and _complex helper.
  - All three public functions now unconditionally call the CPU (_) variants.
  - The CPU (_) variants dispatch to FFT-based convolve/correlate helpers
    (_fft_convolve, _fft_correlate) that work for N-D arrays on both CPU
    (numpy FFT) and GPU (cupy FFT), replacing cupyx.scipy.signal which only
    supports 1D inputs.
  - Two bugs fixed vs the previous FFT implementation:
      1. _fft_convolve no longer discards the imaginary part via .real for
         complex inputs — only real inputs take the .real branch.
      2. _fft_correlate now conjugates b before flipping, matching the
         definition correlate(a,b) = convolve(a, conj(b_flipped)). This
         is a no-op for real inputs but critical for complex k-space data.
  - Everything else is identical to the original.
"""
import numpy as np
import scipy.signal as signal

from sigpy import backend, util

__all__ = ["convolve", "convolve_data_adjoint", "convolve_filter_adjoint"]


# ---------------------------------------------------------------------------
# N-D FFT-based convolve / correlate helpers (CPU and GPU unified)
# ---------------------------------------------------------------------------

def _fft_convolve(a, b, mode, xp):
    """N-D linear convolution of a and b via FFT.

    Works for both real and complex inputs on CPU (xp=numpy) and GPU
    (xp=cupy). Uses zero-padding to full linear convolution size, then
    trims to the requested mode.
    """
    full_shape = tuple(sa + sb - 1 for sa, sb in zip(a.shape, b.shape))
    fa = xp.fft.fftn(a, s=full_shape)
    fb = xp.fft.fftn(b, s=full_shape)
    out_full = xp.fft.ifftn(fa * fb)

    # Only discard imaginary part when both inputs are real-valued.
    # For complex inputs (e.g. k-space data) keeping the imaginary part
    # is essential — discarding it silently corrupts the result.
    if not (xp.iscomplexobj(a) or xp.iscomplexobj(b)):
        out_full = out_full.real

    out_full = out_full.astype(a.dtype)

    if mode == "full":
        return out_full
    elif mode == "valid":
        # Valid region starts at min(sa,sb)-1 in each dimension
        start = tuple(min(sa, sb) - 1 for sa, sb in zip(a.shape, b.shape))
        valid_shape = tuple(abs(sa - sb) + 1 for sa, sb in zip(a.shape, b.shape))
        slc = tuple(slice(st, st + vs) for st, vs in zip(start, valid_shape))
        return out_full[slc]
    else:
        raise ValueError("Invalid mode, got {}".format(mode))


def _fft_correlate(a, b, mode, xp):
    """N-D cross-correlation of a and b via FFT.

    correlate(a, b) = convolve(a, conj(b_flipped))

    The conjugate is essential for complex inputs and is a no-op for real
    inputs (conj of a real array is itself).
    """
    # Conjugate then flip b along all spatial axes
    b_cf = xp.conj(b)
    for ax in range(b_cf.ndim):
        b_cf = xp.flip(b_cf, axis=ax)
    return _fft_convolve(a, b_cf, mode=mode, xp=xp)


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
# Implementations — CPU (scipy) and GPU (cupy FFT) unified
# ---------------------------------------------------------------------------

def _convolve(data, filt, mode="full", strides=None, multi_channel=False):
    xp = backend.get_array_module(data)

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
                if xp is np:
                    output[k, j] += signal.convolve(
                        data[k, i], filt[j, i], mode=mode
                    )[slc]
                else:
                    output[k, j] += _fft_convolve(
                        data[k, i], filt[j, i], mode=mode, xp=xp
                    )[slc]

    if multi_channel:
        return output.reshape(b + (c_o,) + p)
    return output.reshape(b + p)


def _convolve_data_adjoint(
    output, filt, data_shape, mode="full", strides=None, multi_channel=False
):
    xp = backend.get_array_module(output)

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
        adjoint_mode = "full" if all(m_d >= n_d for m_d, n_d in zip(m, n)) else "valid"

    for k in range(B):
        for j in range(c_o):
            for i in range(c_i):
                output_kj[slc] = output[k, j]
                if xp is np:
                    data[k, i] += signal.correlate(
                        output_kj, filt[j, i], mode=adjoint_mode
                    )
                else:
                    data[k, i] += _fft_correlate(
                        output_kj, filt[j, i], mode=adjoint_mode, xp=xp
                    )

    data = data.reshape(data_shape)
    return data


def _convolve_filter_adjoint(
    output, data, filt_shape, mode="full", strides=None, multi_channel=False
):
    xp = backend.get_array_module(data)

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
        adjoint_mode = "valid" if all(m_d >= n_d for m_d, n_d in zip(m, n)) else "full"

    filt = xp.zeros((c_o, c_i) + n, dtype=output.dtype)
    for k in range(B):
        for j in range(c_o):
            for i in range(c_i):
                output_kj[slc] = output[k, j]
                if xp is np:
                    filt[j, i] += signal.correlate(
                        output_kj, data[k, i], mode=adjoint_mode
                    )
                else:
                    filt[j, i] += _fft_correlate(
                        output_kj, data[k, i], mode=adjoint_mode, xp=xp
                    )

    filt = filt.reshape(filt_shape)
    return filt