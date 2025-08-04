import sys
import torch
sys.path.append('/data1/zx/rome-main')
import numpy
import re
from collections import defaultdict
import os
from matplotlib import pyplot as plt
from lm_eval.models.utils import (
    Collator,
    clear_torch_cache,
    configure_pad_token,
    get_dtype,
    pad_and_concat,
    stop_sequences_criteria,
)
from typing import Dict, List, Literal, Optional, Tuple, Union
from util import nethook
import torch.nn.functional as F

def calculate_hidden_flow(
    mt,
    knowledge,
    samples=5,
    noise=0.1,
    token_range=None,
    uniform_noise=False,
    replace=False,
    window=10,
    kind=None,
    expect=None,
    device=None,
    data_erange=None,
):
    """
    Runs causal tracing over every token/layer combination in the network
    and returns a dictionary numerically summarizing the results.
    """
    task = knowledge[-1]
    inp = [knowledge[0] for _ in range((samples + 1))]
    import random
    selected_number = random.randint(0, len(data_erange['number_loc'])-1)
    e_range = data_erange['number_loc'][selected_number] if len(data_erange['number_loc'])>0 else None
    if e_range==None:
        return None
    extracted_tokens = knowledge[0][1][data_erange['question_loc'][0]:data_erange['question_loc'][1]]

    if mt.backend == "causal":
        ntoks_flag = (inp[0][1] + inp[0][2])[-(mt.max_length + 1) :][:-1]
    elif mt.backend == "seq2seq":
        ntoks_flag = inp[0][1][-mt.max_length :]
    input_tokens = [mt.tokenizer.decode([t]) for t in ntoks_flag]
    if token_range == "subject_last":
        token_range = [e_range[1] - 1]
    elif token_range is not None:
        raise ValueError(f"Unknown token_range: {token_range}")
    low_score = trace_with_patch(
        mt, inp, [], knowledge[0][-1], e_range, noise=noise, uniform_noise=uniform_noise
    )
    base_score = trace_with_patch(
        mt, inp, [], knowledge[0][-1], e_range, noise=0., uniform_noise=uniform_noise
    )
    num_layers = count_layers(mt.model)
    print('num_layers: ',num_layers)
    if not kind or kind:
        differences = trace_important_states(
            mt,
            num_layers,
            inp,
            e_range,
            knowledge[0][-1],
            noise=noise,
            kind=kind,
            uniform_noise=uniform_noise,
            replace=replace,
            token_range=token_range,
            data_erange=data_erange,
        )
    else:
        differences = trace_important_window(
            mt.model,
            mt.num_layers,
            inp,
            e_range,
            answer_t,
            noise=noise,
            uniform_noise=uniform_noise,
            replace=replace,
            window=window,
            kind=kind,
            token_range=token_range,
        )
    differences = differences.detach().cpu()
    return dict(
        scores=differences,
        low_score=low_score,
        high_score=base_score,
        input_ids=inp[0][1],
        input_tokens=input_tokens,
        subject_range=(e_range[0]-data_erange['question_loc'][0],e_range[1]-data_erange['question_loc'][0]),
        answer=knowledge[0][0][-1],
        window=window,
        correct_prediction=True,
        kind=kind or "",
    )

def trace_with_patch(
    mt,  # The model
    inp,
    states_to_patch,  # A list of (token index, layername) triples to restore
    answers_t,  # Answer probabilities to collect
    tokens_to_mix,  # Range of tokens to corrupt (begin, end)
    noise=0.1,  # Level of noise to add
    uniform_noise=False,
    replace=False,  # True to replace with instead of add noise
    trace_layers=None,  # List of traced outputs to return
):
    """
    Runs a single causal trace.  Given a model and a batch input where
    the batch size is at least two, runs the batch in inference, corrupting
    a the set of runs [1...n] while also restoring a set of hidden states to
    the values from an uncorrupted run [0] in the batch.

    The convention used by this function is that the zeroth element of the
    batch is the uncorrupted run, and the subsequent elements of the batch
    are the corrupted runs.  The argument tokens_to_mix specifies an
    be corrupted by adding Gaussian noise to the embedding for the batch
    inputs other than the first element in the batch.  Alternately,
    subsequent runs could be corrupted by simply providing different
    input tokens via the passed input batch.

    Then when running, a specified set of hidden states will be uncorrupted
    by restoring their values to the same vector that they had in the
    zeroth uncorrupted run.  This set of hidden states is listed in
    states_to_patch, by listing [(token_index, layername), ...] pairs.
    To trace the effect of just a single state, this can be just a single
    token/layer pair.  To trace the effect of restoring a set of states,
    any number of token indices and layers can be listed.
    """

    rs = numpy.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    if uniform_noise:
        prng = lambda *shape: rs.uniform(-1, 1, shape)
    else:
        prng = lambda *shape: rs.randn(*shape)

    patch_spec = defaultdict(list)
    for t, l in states_to_patch:    # statest_to_path []
        patch_spec[l].append(t)
    embed_layername = layername(mt.model, 0, "embed")    # transformer.wte
    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    if isinstance(noise, float):
        noise_fn = lambda x: noise * x
    else:
        noise_fn = noise

    def patch_rep(x, layer):
        if layer == embed_layername:
            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            if tokens_to_mix is not None:
                b, e = tokens_to_mix
                noise_data = noise_fn(
                    torch.from_numpy(prng(x.shape[0] - 1, e - b, x.shape[2]))
                ).to(x.device)
                if replace:
                    x[1:, b:e] = noise_data
                else:
                    x[1:, b:e] += noise_data
            return x
        if layer not in patch_spec:
            return x
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        h = untuple(x)
        for t in patch_spec[layer]:
            h[1:, t] = h[0, t]
        return x

    # With the patching rules defined, run the patched model in inference.
    # [embed_layername] + list(patch_spec.keys()) + additional_layers === transformer.wte
    additional_layers = [] if trace_layers is None else trace_layers # []
    with torch.no_grad(), nethook.TraceDict(
        mt.model,
        [embed_layername] + list(patch_spec.keys()) + additional_layers,
        edit_output=patch_rep,
    ) as td:
        multi_logits=mt._loglikelihood_tokens(inp, disable_tqdm=True)
    probs = [x[0] for x in multi_logits[1:]]
    probs = sum(probs)/len(probs)
    probs = torch.tensor(probs, dtype=torch.float16)
    return probs

def trace_with_repatch(
    mt,  # The model
    inp,
    states_to_patch,  # A list of (token index, layername) triples to restore
    states_to_unpatch,
    answers_t,  # Answer probabilities to collect
    tokens_to_mix,  # Range of tokens to corrupt (begin, end)
    noise=0.1,  # Level of noise to add
    uniform_noise=False,
    replace=False,  # True to replace with instead of add noise
    trace_layers=None,  # List of traced outputs to return
):
    rs = numpy.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    if uniform_noise:
        prng = lambda *shape: rs.uniform(-1, 1, shape)
    else:
        prng = lambda *shape: rs.randn(*shape)

    patch_spec = defaultdict(list)
    for t, l in states_to_patch:    # statest_to_path []
        patch_spec[l].append(t)
    unpatch_spec = defaultdict(list)
    for t, l in states_to_unpatch:
        unpatch_spec[l].append(t)

    embed_layername = layername(mt.model, 0, "embed")    # transformer.wte
    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    if isinstance(noise, float):
        noise_fn = lambda x: noise * x
    else:
        noise_fn = noise

    def patch_rep(x, layer):
        if layer == embed_layername:
            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            if tokens_to_mix is not None:
                b, e = tokens_to_mix
                noise_data = noise_fn(
                    torch.from_numpy(prng(x.shape[0] - 1, e - b, x.shape[2]))
                ).to(x.device)
                if replace:
                    x[1:, b:e] = noise_data
                else:
                    x[1:, b:e] += noise_data
            return x
        if first_pass or (layer not in patch_spec and layer not in unpatch_spec):
            return x
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        h = untuple(x)
        for t in patch_spec[layer]:
            h[1:, t] = h[0, t]
        for t in unpatch_spec[layer]:
            h[1:, t] = untuple(first_pass_trace[layer].output)[1:, t]
        return x

    # With the patching rules defined, run the patched model in inference.
    for first_pass in [True, False] if states_to_unpatch else [False]:
        with torch.no_grad(), nethook.TraceDict(
            mt.model,
            [embed_layername] + list(patch_spec.keys()) + list(unpatch_spec.keys()),
            edit_output=patch_rep,
        ) as td:
            multi_logits=mt._loglikelihood_tokens(inp, disable_tqdm=True)
            if first_pass:
                first_pass_trace = td
    probs = [x[0] for x in multi_logits[1:]]
    probs = sum(probs)/len(probs)
    probs = torch.tensor(probs, dtype=torch.float16)
    return probs


def layername(model, num, kind=None):
    names = [name for name,module in model.named_modules()]
    if "transformer.h" in names:
        if kind == "embed":
            return "transformer.wte"
        return f'transformer.h.{num}{"" if kind is None else "." + kind}'
    if "model.layers" in names:
        if kind == "embed":
            return [name for name in names if 'embed' in name][0]
        if kind == "attn":
            return f"model.layers.{num}.self_attn"
        if kind == "mlp":
            return f"model.layers.{num}.mlp"
        if kind == None:
            return f"model.layers.{num}"
    if "gpt_neox" in names:
        if kind == "embed":
            return "gpt_neox.embed_in"
        if kind == "attn":
            kind = "attention"
        return f'gpt_neox.layers.{num}{"" if kind is None else "." + kind}'
    assert False, "unknown transformer structure"



def trace_important_states(
    mt,
    num_layers,
    inp,
    e_range,
    answer_t,
    noise=0.1,
    kind=None,
    uniform_noise=False,
    replace=False,
    token_range=None,
    data_erange=None,
):
    if mt.backend == "causal":
        ntoks = len((inp[0][1] + inp[0][2])[-(mt.max_length + 1) :][:-1])
    elif mt.backend == "seq2seq":
        ntoks = len((inp[0][1])[-mt.max_length :])
    else:
        assert False, "unknown transformer structure"
    table = []  
    if token_range is None:
        token_range = range(ntoks)
    for tnum in range(data_erange['question_loc'][0],data_erange['question_loc'][1]):
        row = []
        for layer in range(num_layers):
            r = trace_with_patch(
                mt,
                inp,
                [(tnum, layername(mt.model, layer, kind))],
                answer_t,
                tokens_to_mix=e_range,
                noise=noise,
                uniform_noise=uniform_noise,
                replace=replace,
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)

def trace_important_states_frozen(
    mt,
    num_layers,
    inp,
    e_range,
    answer_t,
    noise=0.1,
    kind=None,
    uniform_noise=False,
    replace=False,
    token_range=None,
    data_erange=None,
):
    if mt.backend == "causal":
        ntoks = len((inp[0][1] + inp[0][2])[-(mt.max_length + 1) :][:-1])
    elif mt.backend == "seq2seq":
        ntoks = len((inp[0][1])[-mt.max_length :])
    else:
        assert False, "unknown transformer structure"
    table = []
    if token_range is None:
        token_range = range(ntoks)
    zero_mlps=[]
    for tnum in range(data_erange['question_loc'][0],data_erange['question_loc'][1]):
        zero_mlps=[]
        if kind=='attn':
            zero_mlps=[(tnum, layername(mt.model, L, "attn")) for L in range(0, num_layers)]
        elif kind=='mlp':
            zero_mlps=[(tnum, layername(mt.model, L, "mlp")) for L in range(0, num_layers)]
        row = []
        for layer in range(num_layers):
            r = trace_with_repatch(
                mt,
                inp,
                [(tnum, layername(mt.model, layer, None))],
                zero_mlps,
                answer_t,
                tokens_to_mix=e_range,
                noise=noise,
                uniform_noise=uniform_noise,
                replace=replace,
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)


def count_layers(model):
    names = [name for name,module in model.named_modules()]
    if "transformer.h" in names:
        layernames=[
            n
            for n, m in model.named_modules()
            if (re.match(r"^(transformer|gpt_neox)\.(h|layers)\.\d+$", n))
        ]
        return len(layernames)
    if "model.layers" in names:
        layernames=[
            n
            for n, m in model.named_modules()
            if (re.match(r"^model.layers.\d+$", n))
        ]
        return len(layernames)

    assert False, "unknown model structure"


def decode_tokens(mt, knowledge):
    #if mt.backend == "causal" or mt.backend =='seq2seq':
        #token_array = (knownledge[1])[-mt.max_length :]
    if mt.backend == "causal":
        token_array = (knowledge[1] + knowledge[2])[-(mt.max_length + 1) :][:-1]
    elif mt.backend == 'seq2seq':
        token_array = knowledge[1][-self.max_length :]
    else:
        assert False, "unknown backend"
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(mt, row) for row in token_array]
    for t in token_array:
        a = mt.tokenizer.decode(t)
    return [mt.tokenizer.decode([t]) for t in token_array]


def plot_trace_heatmap(result, savepdf=None, title=None, xlabel=None, modelname=None):
    differences = result["scores"]
    print(differences.shape)
    low_score = result["low_score"]
    answer = result["answer"]
    kind = (
        None
        if (not result["kind"] or result["kind"] == "None")
        else str(result["kind"])
    )
    print(f'kind:{kind} low_score:{low_score} base_score:{result["high_score"]} \nscores:{differences}')
    window = result.get("window", 10)
    labels = list(result["input_tokens"])
    for i in range(*result["subject_range"]):
        labels[i] = labels[i] + "*"
    with plt.rc_context(rc={"font.family": "Times New Roman"}):
        fig, ax = plt.subplots(figsize=(3.5, 6), dpi=200)
        h = ax.pcolor(
            differences,
            cmap={None: "Purples", "None": "Purples", "mlp": "Greens", "attn": "Reds"}[
                kind
            ],
            vmin=low_score,
        )
        ax.invert_yaxis()
        ax.set_yticks([0.5 + i for i in range(len(differences))])
        ax.set_xticks([0.5 + i for i in range(0, differences.shape[1] - 6, 5)])
        ax.set_xticklabels(list(range(0, differences.shape[1] - 6, 5)))
        ax.set_yticklabels(labels)
        if not modelname:
            modelname = "GPT"
        if not kind:
            ax.set_title("Impact of restoring state after corrupted input")
            ax.set_xlabel(f"single restored layer within {modelname}")
        else:
            kindname = "MLP" if kind == "mlp" else "Attn"
            ax.set_title(f"Impact of restoring {kindname} after corrupted input")
            ax.set_xlabel(f"center of interval of {window} restored {kindname} layers")
        cb = plt.colorbar(h)
        if title is not None:
            ax.set_title(title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        elif answer is not None:
            # The following should be cb.ax.set_xlabel, but this is broken in matplotlib 3.5.1.
            cb.ax.set_title(f"p({str(answer).strip()})", y=-0.16, fontsize=10)
        if savepdf:
            os.makedirs(os.path.dirname(savepdf), exist_ok=True)
            plt.savefig(savepdf, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


def collect_embedding_std(mt, new_reqs):
    alldata = []
    for s in new_reqs:
        with torch.no_grad(), nethook.Trace(mt.model, layername(mt.model, 0, "embed")) as t:
            mt._loglikelihood_tokens([s], disable_tqdm=True)
            alldata.append(t.output[0].cpu())
    alldata = torch.cat(alldata)
    noise_level = alldata.std().item()
    return noise_level
