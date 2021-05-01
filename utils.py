import random
import numpy
import numpy as np
from PIL import Image
from PIL.Image import BICUBIC
import logging

def batchfy_by_frame(
    sorted_data,
    max_frames_in,
    max_frames_out,
    max_frames_inout,
    num_batches=0,
    min_batch_size=1,
    shortest_first=False,
    ikey="input",
    okey="output",
):
    if max_frames_in <= 0 and max_frames_out <= 0 and max_frames_inout <= 0:
        raise ValueError(
            "At least, one of `--batch-frames-in`, `--batch-frames-out` or "
            "`--batch-frames-inout` should be > 0"
        )
    length = len(sorted_data)
    minibatches = []
    start = 0
    end = 0
    while end != length:
        # Dynamic batch size depending on size of samples
        b = 0
        max_olen = 0
        max_ilen = 0
        while (start + b) < length:
            ilen = int(sorted_data[start + b][1][ikey][0]["shape"][0])
            if ilen > max_frames_in and max_frames_in != 0:
                raise ValueError(
                    f"Can't fit one sample in --batch-frames-in ({max_frames_in}): "
                    f"Please increase the value"
                )
            olen = int(sorted_data[start + b][1][okey][0]["shape"][0])
            if olen > max_frames_out and max_frames_out != 0:
                raise ValueError(
                    f"Can't fit one sample in --batch-frames-out ({max_frames_out}): "
                    f"Please increase the value"
                )
            if ilen + olen > max_frames_inout and max_frames_inout != 0:
                raise ValueError(
                    f"Can't fit one sample in --batch-frames-out ({max_frames_inout}): "
                    f"Please increase the value"
                )
            max_olen = max(max_olen, olen)
            max_ilen = max(max_ilen, ilen)
            in_ok = max_ilen * (b + 1) <= max_frames_in or max_frames_in == 0
            out_ok = max_olen * (b + 1) <= max_frames_out or max_frames_out == 0
            inout_ok = (max_ilen + max_olen) * (
                b + 1
            ) <= max_frames_inout or max_frames_inout == 0
            if in_ok and out_ok and inout_ok:
                # add more seq in the minibatch
                b += 1
            else:
                # no more seq in the minibatch
                break
        end = min(length, start + b)
        batch = sorted_data[start:end]
        if shortest_first:
            batch.reverse()
        minibatches.append(batch)
        # Check for min_batch_size and fixes the batches if needed
        i = -1
        while len(minibatches[i]) < min_batch_size:
            missing = min_batch_size - len(minibatches[i])
            if -i == len(minibatches):
                minibatches[i + 1].extend(minibatches[i])
                minibatches = minibatches[1:]
                break
            else:
                minibatches[i].extend(minibatches[i - 1][:missing])
                minibatches[i - 1] = minibatches[i - 1][missing:]
                i -= 1
        start = end
    if num_batches > 0:
        minibatches = minibatches[:num_batches]
    lengths = [len(x) for x in minibatches]
    logging.info(
        str(len(minibatches))
        + " batches containing from "
        + str(min(lengths))
        + " to "
        + str(max(lengths))
        + " samples"
        + "(avg "
        + str(int(np.mean(lengths)))
        + " samples)."
    )

    return minibatches
BATCH_COUNT_CHOICES = ["auto", "seq", "bin", "frame"]
BATCH_SORT_KEY_CHOICES = ["input", "output", "shuffle"]

def make_batchset(
    data,
    batch_size=0,
    max_length_in=float("inf"),
    max_length_out=float("inf"),
    num_batches=0,
    min_batch_size=1,
    shortest_first=False,
    batch_sort_key="input",
    swap_io=False,
    mt=False,
    count="auto",
    batch_bins=0,
    batch_frames_in=0,
    batch_frames_out=0,
    batch_frames_inout=0,
    iaxis=0,
    oaxis=0,
):

    # check args
    if count not in BATCH_COUNT_CHOICES:
        raise ValueError(
            f"arg 'count' ({count}) should be one of {BATCH_COUNT_CHOICES}"
        )
    if batch_sort_key not in BATCH_SORT_KEY_CHOICES:
        raise ValueError(
            f"arg 'batch_sort_key' ({batch_sort_key}) should be "
            f"one of {BATCH_SORT_KEY_CHOICES}"
        )

    # TODO(karita): remove this by creating converter from ASR to TTS json format
    batch_sort_axis = 0
    if swap_io:
        # for TTS
        ikey = "output"
        okey = "input"
        if batch_sort_key == "input":
            batch_sort_key = "output"
        elif batch_sort_key == "output":
            batch_sort_key = "input"
    elif mt:
        # for MT
        ikey = "output"
        okey = "output"
        batch_sort_key = "output"
        batch_sort_axis = 1
        assert iaxis == 1
        assert oaxis == 0
        # NOTE: input is json['output'][1] and output is json['output'][0]
    else:
        ikey = "input"
        okey = "output"

    if count == "auto":
        if batch_size != 0:
            count = "seq"
        elif batch_bins != 0:
            count = "bin"
        elif batch_frames_in != 0 or batch_frames_out != 0 or batch_frames_inout != 0:
            count = "frame"
        else:
            raise ValueError(
                f"cannot detect `count` manually set one of {BATCH_COUNT_CHOICES}"
            )
        logging.info(f"count is auto detected as {count}")

    if count != "seq" and batch_sort_key == "shuffle":
        raise ValueError("batch_sort_key=shuffle is only available if batch_count=seq")

    category2data = {}  # Dict[str, dict]
    for k, v in data.items():
        category2data.setdefault(v.get("category"), {})[k] = v

    batches_list = []  # List[List[List[Tuple[str, dict]]]]
    for d in category2data.values():
        if batch_sort_key == "shuffle":
            batches = batchfy_shuffle(
                d, batch_size, min_batch_size, num_batches, shortest_first
            )
            batches_list.append(batches)
            continue

        # sort it by input lengths (long to short)
        sorted_data = sorted(
            d.items(),
            key=lambda data: int(data[1][batch_sort_key][batch_sort_axis]["shape"][0]),
            reverse=not shortest_first,
        )
        logging.info("# utts: " + str(len(sorted_data)))
        if count == "seq":
            raise ValueError(f"Error")
        if count == "bin":
            raise ValueError(f"Error")
        if count == "frame":
            batches = batchfy_by_frame(
                sorted_data,
                max_frames_in=batch_frames_in,
                max_frames_out=batch_frames_out,
                max_frames_inout=batch_frames_inout,
                min_batch_size=min_batch_size,
                shortest_first=shortest_first,
                ikey=ikey,
                okey=okey,
            )
        batches_list.append(batches)

    if len(batches_list) == 1:
        batches = batches_list[0]
    else:
        # Concat list. This way is faster than "sum(batch_list, [])"
        batches = list(itertools.chain(*batches_list))

    # for debugging
    if num_batches > 0:
        batches = batches[:num_batches]
    logging.info("# minibatches: " + str(len(batches)))

    # batch: List[List[Tuple[str, dict]]]
    return batches

#SpecAugment
def time_warp(x, max_time_warp=80, inplace=False, mode="PIL"):
    """time warp for spec augment
    move random center frame by the random width ~ uniform(-window, window)
    :param numpy.ndarray x: spectrogram (time, freq)
    :param int max_time_warp: maximum time frames to warp
    :param bool inplace: overwrite x with the result
    :param str mode: "PIL" (default, fast, not differentiable) or "sparse_image_warp"
        (slow, differentiable)
    :returns numpy.ndarray: time warped spectrogram (time, freq)
    """
    window = max_time_warp
    if mode == "PIL":
        t = x.shape[0]
        if t - window <= window:
            return x
        # NOTE: randrange(a, b) emits a, a + 1, ..., b - 1
        center = random.randrange(window, t - window)
        warped = random.randrange(center - window, center + window) + 1  # 1 ... t - 1

        left = Image.fromarray(x[:center]).resize((x.shape[1], warped), BICUBIC)
        right = Image.fromarray(x[center:]).resize((x.shape[1], t - warped), BICUBIC)
        if inplace:
            x[:warped] = left
            x[warped:] = right
            return x
        return numpy.concatenate((left, right), 0)
    elif mode == "sparse_image_warp":
        import torch

        from espnet.utils import spec_augment

        # TODO(karita): make this differentiable again
        return spec_augment.time_warp(torch.from_numpy(x), window).numpy()
    else:
        raise NotImplementedError(
            "unknown resize mode: "
            + mode
            + ", choose one from (PIL, sparse_image_warp)."
        )


def freq_mask(x, F=30, n_mask=2, replace_with_zero=True, inplace=False):
    """freq mask for spec agument
    :param numpy.ndarray x: (time, freq)
    :param int n_mask: the number of masks
    :param bool inplace: overwrite
    :param bool replace_with_zero: pad zero on mask if true else use mean
    """
    if inplace:
        cloned = x
    else:
        cloned = x.copy()

    num_mel_channels = cloned.shape[1]
    fs = numpy.random.randint(0, F, size=(n_mask, 2))

    for f, mask_end in fs:
        f_zero = random.randrange(0, num_mel_channels - f)
        mask_end += f_zero

        # avoids randrange error if values are equal and range is empty
        if f_zero == f_zero + f:
            continue

        if replace_with_zero:
            cloned[:, f_zero:mask_end] = 0
        else:
            cloned[:, f_zero:mask_end] = cloned.mean()
    return cloned




def time_mask(spec, T=40, n_mask=2, replace_with_zero=True, inplace=False):
    """freq mask for spec agument
    :param numpy.ndarray spec: (time, freq)
    :param int n_mask: the number of masks
    :param bool inplace: overwrite
    :param bool replace_with_zero: pad zero on mask if true else use mean
    """
    if inplace:
        cloned = spec
    else:
        cloned = spec.copy()
    len_spectro = cloned.shape[0]
    ts = numpy.random.randint(0, T, size=(n_mask, 2))
    for t, mask_end in ts:
        # avoid randint range error
        if len_spectro - t <= 0:
            continue
        t_zero = random.randrange(0, len_spectro - t)

        # avoids randrange error if values are equal and range is empty
        if t_zero == t_zero + t:
            continue

        mask_end += t_zero
        if replace_with_zero:
            cloned[t_zero:mask_end] = 0
        else:
            cloned[t_zero:mask_end] = cloned.mean()
    return cloned


def spec_augment(
    x,
    resize_mode="PIL",
    max_time_warp=5,
    max_freq_width=30,
    n_freq_mask=2,
    max_time_width=40,
    n_time_mask=2,
    inplace=True,
    replace_with_zero=False,
):
    """spec agument
    apply random time warping and time/freq masking
    default setting is based on LD (Librispeech double) in Table 2
        https://arxiv.org/pdf/1904.08779.pdf
    :param numpy.ndarray x: (time, freq)
    :param str resize_mode: "PIL" (fast, nondifferentiable) or "sparse_image_warp"
        (slow, differentiable)
    :param int max_time_warp: maximum frames to warp the center frame in spectrogram (W)
    :param int freq_mask_width: maximum width of the random freq mask (F)
    :param int n_freq_mask: the number of the random freq mask (m_F)
    :param int time_mask_width: maximum width of the random time mask (T)
    :param int n_time_mask: the number of the random time mask (m_T)
    :param bool inplace: overwrite intermediate array
    :param bool replace_with_zero: pad zero on mask if true else use mean
    """
    assert isinstance(x, numpy.ndarray)
    assert x.ndim == 2
    x = time_warp(x, max_time_warp, inplace=inplace, mode=resize_mode)
    x = freq_mask(
        x,
        max_freq_width,
        n_freq_mask,
        inplace=inplace,
        replace_with_zero=replace_with_zero,
    )
    x = time_mask(
        x,
        max_time_width,
        n_time_mask,
        inplace=inplace,
        replace_with_zero=replace_with_zero,
    )
    return x
