import functools
import torch

from quantizers.boa import BoA
from quantizers.minmax import MinMaxQuantizer
from utils.model_utils import get_transformer_blocks, get_head_info, get_rotary_emb, cache_first_transformer_input
from utils.utils import find_layers, cleanup_memory

QKV_NAMES = {"query": "self_attn.q_proj", "key": "self_attn.k_proj", "value": "self_attn.v_proj"}


@torch.no_grad()
def boa_fwrd(llm, calib_data, qconfigs, boa_opts: dict, hyperparams: dict, args):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_cache = llm.config.use_cache
    llm.config.use_cache = False

    # cache inputs for fast quantization
    quant_inps, block_kwargs = cache_first_transformer_input(llm, calib_data)

    transformer_blocks = get_transformer_blocks(llm)
    n_heads, n_kv_heads, head_dim = get_head_info(llm)
    rotary_emb = get_rotary_emb(llm)
    rotary_matrix = get_rotary_matrix(rotary_emb, llm.config, block_kwargs['position_ids'].cpu()) if rotary_emb is not None else None    
    dump_deltas = getattr(args, 'dump_deltas', False)
    if dump_deltas:
        from diag.dump_utils import default_dump_dir, write_manifest, save_delta
        dump_dir = args.dump_dir or default_dump_dir(args)
        write_manifest(dump_dir, args)
        print(f'[dump_deltas] writing per-layer dW to {dump_dir}')

    phase1 = getattr(args, 'phase1', False)
    if phase1:
        from diag.phase1_runner import Phase1Collector
        collector = Phase1Collector(args.phase1_dir, n_calib=args.phase1_ncalib,
                                    want_attn=not args.phase1_no_attn, seed=args.seed)
        print(f'[phase1] Kronecker-gap diagnostic -> {args.phase1_dir}')

    phase4 = getattr(args, 'phase4', False)
    if phase4:
        from diag.phase4_fc1 import Phase4Collector
        collector4 = Phase4Collector(args.phase4_dir, tokens_per_seq=args.phase4_tokens_per_seq,
                                     seed=args.seed)
        print(f'[phase4] fc1/ReLU Kronecker-gap diagnostic -> {args.phase4_dir}')

    dense = getattr(args, 'dense_arm', 'none') != 'none' or getattr(args, 'obj_eval', False)
    if dense:
        from diag.dense_driver import DenseCollector
        from utils.data_utils import get_calib_data
        import copy
        ho_args = copy.copy(args); ho_args.seed = 1000 + args.seed; ho_args.nsamples = args.heldout_nsamples
        heldout_inps, _ = cache_first_transformer_input(llm, get_calib_data(ho_args))
        dcoll = DenseCollector(args.dense_arm, [int(b) for b in args.dense_blocks.split(',')],
                               args.dense_dir, heldout_inps, n_dense_seqs=args.dense_nsamples,
                               tokens_per_seq=args.dense_tokens_per_seq, seed=args.seed,
                               obj_eval=args.obj_eval or args.dense_arm != 'none')
        print(f"[dense] arm={args.dense_arm} blocks={sorted(dcoll.blocks)} -> {args.dense_dir}")

    # quantize each Transformer block
    for i in range(len(transformer_blocks)):
        print(f'>>>> Quantizing {i+1}-th Transformer Block.... ({i+1}/{len(transformer_blocks)})')
        transformer_block = transformer_blocks[i].to(dev)
        
        fp_layers = find_layers(transformer_block)

        wrappers = {}
        for name, fp_layer in fp_layers.items():
            wrappers[name] = BoA(fp_layer, boa_opts, hyperparams)
            wrappers[name].quantizer = MinMaxQuantizer()
            wrappers[name].quantizer.configure(qconfigs["w_bits"], per_channel=True, sym=qconfigs["w_sym"], mse=False)
            wrappers[name].quantizer.find_params(wrappers[name].layer.weight.data)

        # compute Hessians
        block_v = boa_opts['block_v']
        compute_Hessian(transformer_block, n_heads, n_kv_heads, head_dim, wrappers, quant_inps, block_kwargs, block_v, rotary_matrix,
                        row_metric_v=boa_opts.get('row_metric_v', False),
                        rsq=boa_opts if boa_opts.get('rsq_weights') else None,
                        row_metric_fc1=boa_opts.get('row_metric_fc1_groups', 0) if boa_opts.get('row_metric_fc1') else False,
                        q_centered=boa_opts.get('q_centered', False),
                        q_identity=boa_opts.get('q_identity', False))

        if phase1:
            collector.on_block_hessians(transformer_block, i, wrappers, quant_inps,
                                        block_kwargs, n_heads, head_dim)
        if phase4:
            collector4.on_block_hessians(transformer_block, i, wrappers, quant_inps, block_kwargs)
        if dense:
            dcoll.on_block_start(transformer_block, i, quant_inps, block_kwargs, n_heads, head_dim)

        # quantize
        need_W_orig = dump_deltas or phase1 or phase4 or dense

        # --qk_quantK: quantize the key projection first so q_proj's row metric can be
        # rebuilt against the quantized K (see requantized_key_row_metric).
        qk_quant_k = boa_opts.get('qk_quant_k', False)
        layer_order = list(fp_layers)
        if qk_quant_k:
            q_nm, k_nm = QKV_NAMES["query"], QKV_NAMES["key"]
            if q_nm in layer_order and k_nm in layer_order \
                    and layer_order.index(k_nm) > layer_order.index(q_nm):
                layer_order.remove(k_nm)
                layer_order.insert(layer_order.index(q_nm), k_nm)

        for name in layer_order:
            print('-' * 50)
            print(f">>> Layer: {name}")
            W_orig = fp_layers[name].weight.data.clone() if need_W_orig else None
            if not (dense and dcoll.quantize_layer(i, name, wrappers[name], boa_opts)):
                wrappers[name].quant(args.print_memory_usage)
            if dense:
                dcoll.on_layer_done(i, name, W_orig, fp_layers[name].weight.data)
            if dump_deltas:
                save_delta(dump_dir, i, name, W_orig, fp_layers[name].weight.data)
            if phase1:
                collector.on_layer_quantized(i, name, W_orig, fp_layers[name].weight.data)
            if phase4:
                collector4.on_layer_quantized(i, name, W_orig, fp_layers[name].weight.data)
            if qk_quant_k and name == QKV_NAMES["key"]:
                wrappers[QKV_NAMES["query"]].H_row = requantized_key_row_metric(
                    transformer_block, n_heads, n_kv_heads, head_dim,
                    quant_inps, block_kwargs, rotary_matrix)

            del W_orig
            wrappers[name].free()

        if phase1:
            collector.finish_block(i)
        if phase4:
            collector4.finish_block(i)
        if dense:
            dcoll.on_block_end(transformer_block, i)

        # cache inputs for next transformer block
        for j in range(len(quant_inps)):
            quant_inps[j] = transformer_block(quant_inps[j].unsqueeze(0), **block_kwargs)[0]
        
        transformer_blocks[i] = transformer_block.cpu()
        del transformer_block
        del wrappers 
        
        cleanup_memory(verbose=False)

    llm.config.use_cache = use_cache


def get_rotary_matrix(rotary_emb, config, position_ids):
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    half_head_dim = head_dim // 2
    seqlen = position_ids.shape[-1]

    cos, sin = rotary_emb(torch.rand([1], dtype=torch.float32), position_ids=position_ids)
    cos, sin = cos.squeeze(), sin.squeeze()

    rotary_matrix = torch.zeros(*(seqlen, head_dim, head_dim), dtype=cos.dtype, device=cos.device)
    rotary_matrix[:, :half_head_dim, :half_head_dim] = torch.diag_embed(cos[:, :half_head_dim])
    rotary_matrix[:, :half_head_dim, half_head_dim:] = -torch.diag_embed(sin[:, :half_head_dim])
    rotary_matrix[:, half_head_dim:, :half_head_dim] = torch.diag_embed(sin[:, :half_head_dim])
    rotary_matrix[:, half_head_dim:, half_head_dim:] = torch.diag_embed(cos[:, :half_head_dim])

    rotary_matrix = rotary_matrix.unsqueeze(dim=1)
    return rotary_matrix


@torch.no_grad()
def requantized_key_row_metric(transformer_block, n_heads, n_kv_heads, head_dim,
                               quant_inps, block_kwargs, rotary_matrix):
    """q_proj's output metric, re-measured through the ALREADY-QUANTIZED k_proj.

    BoA builds q_proj's row metric from E[K K^T] on the FP key projection, but at
    inference the logits are formed against the quantized K. --qk_quantK closes that
    mismatch: k_proj is quantized first, then this re-measures the post-RoPE key
    covariance with the quantized k_proj in place and hands it back as q_proj's
    H_row. Everything else (H_col, the solver, k_proj's own metric) is untouched.

    Costs one extra block forward over the calibration set; attentions are not
    requested, so it is cheaper than the pass in compute_Hessian.

    Returns [n_heads, head_dim, head_dim].
    """
    from utils.hessian_utils import CovarianceCollector

    target = (transformer_block.self_attn.rot_out_K if rotary_matrix is not None
              else find_layers(transformer_block)[QKV_NAMES["key"]])
    coll = CovarianceCollector(target)
    handle = target.register_forward_hook(
        functools.partial(coll.compute_cov_out_batch, n_heads=n_kv_heads))

    kw = {k: v for k, v in block_kwargs.items() if k != 'output_attentions'}
    for j in range(len(quant_inps)):
        transformer_block(quant_inps[j].unsqueeze(0), **kw)
    handle.remove()

    YYT = coll.YYT
    if n_kv_heads != n_heads:                       # expand kv heads to query heads
        n_shared = n_heads // n_kv_heads
        YYT = YYT[:, None, :, :].expand(-1, n_shared, -1, -1).reshape(n_heads, head_dim, head_dim)
    del coll
    if rotary_matrix is None:
        return YYT
    R = rotary_matrix.cuda()
    return (R.transpose(-1, -2) @ YYT @ R).mean(0)


def get_out_proj(transformer_block):
    """Attention output projection: OPT names it out_proj, Llama/Qwen name it o_proj."""
    attn = transformer_block.self_attn
    for nm in ("out_proj", "o_proj"):
        if hasattr(attn, nm):
            return getattr(attn, nm)
    raise AttributeError(f"no output projection on {type(attn).__name__}")


def value_row_metric(transformer_block, n_heads, n_kv_heads, head_dim, device, dtype=torch.float32):
    """Row (output-side) Hessian factor for the value projection -- paper eq. (9).

        H(w_V,h) = 2 X A_h^T A_h X^T  (x)  W_out,h^T W_out,h

    The released code builds only the left factor (that is what --block_v is) and
    leaves H_row unset, so v_proj falls through to the one-sided gptq() path with an
    identity output metric. Unlike the q/k Hessians -- which come from a Taylor step
    and a Cauchy-Schwarz relaxation (paper sec. 3.3) -- eq. (9) is EXACT: when only
    V changes, A_h does not move, so ||dMHA||_F^2 = ||W_out,h dW_V,h X A_h^T||_F^2
    with no approximation. W_out,h is deterministic, so the Kronecker form is exact
    too; no expectation has to factorise.

    Under GQA (n_kv_heads < n_heads) v_proj emits only n_kv_heads * head_dim rows,
    and each kv head g is read by n_shared = n_heads // n_kv_heads query heads. With
    the other heads held fixed, dMHA = sum_h W_out,h dV_{g(h)} A_h^T, so the row
    factor for kv head g is the SUM of W_out,h^T W_out,h over the query heads in its
    group -- not a mean, and not one factor per query head. (For MHA n_shared = 1 and
    this reduces to the original per-head expression, so the OPT path is unchanged.)

    Returns [n_kv_heads, head_dim, head_dim].
    """
    W_o = get_out_proj(transformer_block).weight.data.to(device=device, dtype=dtype)
    n_shared = n_heads // n_kv_heads
    H_row = torch.zeros(n_kv_heads, head_dim, head_dim, device=device, dtype=dtype)
    for h in range(n_heads):
        W_oh = W_o[:, h * head_dim:(h + 1) * head_dim]        # [d, d_h]
        H_row[h // n_shared] += W_oh.transpose(-1, -2) @ W_oh
    return H_row


class MeanCollector:
    """Running mean of a layer's per-head output, matching CovarianceCollector's
    preprocessing. Needed to CENTRE the key second moment: compute_cov accumulates
    YYT = 2 * E[k k^T], so the centred form is YYT - 2 * mu mu^T."""

    def __init__(self):
        self.S = None
        self.n = 0

    def hook(self, _, inp, out, n_heads=None):
        from utils.hessian_utils import preprocess
        d = preprocess(out, n_heads)                    # [H, d_h, BL]
        s = d.double().sum(-1)
        self.S = s if self.S is None else self.S + s
        self.n += d.shape[-1]

    def mean(self):
        return self.S / max(self.n, 1)


class ReluCoactivation:
    """Accumulates E_t[d_t d_t^T] from the ReLU mask d_t = 1[fc1(x_t) > 0] via a hook
    on fc1's OUTPUT (pre-activation, bias included). Phase 4 row metric for fc1:

        H_row_fc1 = W2^T W2 * E_t[d d^T]      (Hadamard product)   -- exact per token,
                                                pooled over tokens (Kronecker level)
    """
    def __init__(self, d_ff, device):
        self.C = torch.zeros(d_ff, d_ff, device=device)
        self.n = 0

    def hook(self, _, inp, out):
        D = (out.detach().reshape(-1, out.shape[-1]) > 0).float()      # [T, d_ff]
        self.C += D.T @ D
        self.n += D.shape[0]


@torch.no_grad()
def compute_Hessian(transformer_block, n_heads, n_kv_heads, head_dim, wrappers, quant_inps, block_kwargs, block_v, rotary_matrix, row_metric_v=False, rsq=None, row_metric_fc1=False, q_centered=False, q_identity=False):
    from utils.hessian_utils import CovarianceCollector, preprocess, compute_cov

    layers = find_layers(transformer_block)
    cov_collectors = {}
    for name, layer in layers.items():
        cov_collectors[name] = CovarianceCollector(layer)
    if rotary_matrix is not None:  # For models exploiting RoPE, we need to save the covariance of outputs after RoPE.
        cov_collectors['rot_out_Q'] = CovarianceCollector(transformer_block.self_attn.rot_out_Q)
        cov_collectors['rot_out_K'] = CovarianceCollector(transformer_block.self_attn.rot_out_K)

    handles = []
    for name in layers:
        if name in [QKV_NAMES["query"], QKV_NAMES["key"]]:  # Q, K, V share inputs.
            pass
        elif name == QKV_NAMES["value"]:
            handles.append(layers[name].register_forward_hook(cov_collectors[name].compute_cov_in_batch))
            if block_v:
                handles.append(layers[name].register_forward_hook(cov_collectors[name].save_inps))  # we need to compute XATAXT for value
        else:
            handles.append(layers[name].register_forward_hook(cov_collectors[name].compute_cov_in_batch))

    if rotary_matrix is None:
        handles.append(layers[QKV_NAMES["query"]].register_forward_hook(functools.partial(cov_collectors[QKV_NAMES["query"]].compute_cov_out_batch, n_heads=n_heads)))
        handles.append(layers[QKV_NAMES["key"]].register_forward_hook(functools.partial(cov_collectors[QKV_NAMES["key"]].compute_cov_out_batch, n_heads=n_kv_heads)))
    else:
        handles.append(transformer_block.self_attn.rot_out_Q.register_forward_hook(functools.partial(cov_collectors['rot_out_Q'].compute_cov_out_batch, n_heads=n_heads)))
        handles.append(transformer_block.self_attn.rot_out_K.register_forward_hook(functools.partial(cov_collectors['rot_out_K'].compute_cov_out_batch, n_heads=n_kv_heads)))
    kmean = None
    if q_centered:
        kmean = MeanCollector()
        _tgt = (transformer_block.self_attn.rot_out_K if rotary_matrix is not None
                else layers[QKV_NAMES["key"]])
        handles.append(_tgt.register_forward_hook(
            functools.partial(kmean.hook, n_heads=n_kv_heads)))

    if row_metric_fc1:
        coact = ReluCoactivation(layers['fc1'].out_features, layers['fc1'].weight.device)
        handles.append(layers['fc1'].register_forward_hook(coact.hook))

    if block_v:
        block_kwargs = block_kwargs.copy()
        block_kwargs['output_attentions'] = True
        XXT_value, n_data_in_value = 0, 0

    # RSQ token weights (diag/rsq_weights.py): a second, weighted running covariance
    # of the shared q/k/v input (and, with rsq_all_layers, of every layer's input).
    if rsq is not None and rsq.get('rsq_weights'):
        from diag.rsq_weights import attention_concentration, weighted_cov_update
        from utils.hessian_utils import preprocess as _pp
        rsq_cov = {}
        rsq_layers = list(layers) if rsq.get('rsq_all_layers') else [QKV_NAMES["value"]]
        for name in rsq_layers:
            rsq_cov[name] = [0, 0]
            handles.append(layers[name].register_forward_hook(
                (lambda nm: lambda m, i, o: rsq_cov[nm].append(_pp(i[0].data)))(name)))
        block_kwargs = block_kwargs.copy()
        block_kwargs['output_attentions'] = True
    
    for j in range(len(quant_inps)):
        if rsq is not None and rsq.get('rsq_weights'):
            out = transformer_block(quant_inps[j].unsqueeze(0), **block_kwargs)
            A = out[-1][0]                                          # [M, T, T]
            r = attention_concentration(A, r_min=rsq['rsq_min_value'])
            for name in rsq_layers:
                Xin = rsq_cov[name].pop()                           # [d, T] captured by hook
                rsq_cov[name][0], rsq_cov[name][1] = weighted_cov_update(rsq_cov[name][0], rsq_cov[name][1], Xin, r)
            if block_v:
                quant_A = out[-1]
                quant_inp_value = cov_collectors[QKV_NAMES["value"]].quant_inp.pop()
                quant_inp_value = torch.einsum('bhli, bid -> bhld', quant_A, quant_inp_value)
                quant_inp_value = preprocess(quant_inp_value, n_heads)
                XXT_value, n_data_in_value = compute_cov(XXT_value, n_data_in_value, quant_inp_value)
            continue
        if not block_v:
            transformer_block(quant_inps[j].unsqueeze(0), **block_kwargs)
        else:
            quant_A = transformer_block(quant_inps[j].unsqueeze(0), **block_kwargs)[-1]
            quant_inp_value = cov_collectors[QKV_NAMES["value"]].quant_inp.pop()
            quant_inp_value = torch.einsum('bhli, bid -> bhld', quant_A, quant_inp_value)
            quant_inp_value = preprocess(quant_inp_value, n_heads)
    
            XXT_value, n_data_in_value = compute_cov(XXT_value, n_data_in_value, quant_inp_value)
            
    for h in handles:
        h.remove()
    
    # Assign H_col except for value
    for name, wrapper in wrappers.items():
        if name in [QKV_NAMES["query"], QKV_NAMES["key"]]:
            wrapper.H_col = cov_collectors[QKV_NAMES["value"]].XXT
        elif name == QKV_NAMES["value"]:
            pass
        else:
            wrapper.H_col = cov_collectors[name].XXT

    # Assign H_col for value
    if block_v:
        if n_kv_heads != n_heads:
            n_shared = n_heads // n_kv_heads
            hidden_size = XXT_value.shape[-1]
            XXT_value = XXT_value.reshape(n_kv_heads, n_shared, hidden_size, hidden_size).mean(dim=1)        
        wrappers[QKV_NAMES['value']].H_col = XXT_value
        del XXT_value

    else:
        wrappers[QKV_NAMES['value']].H_col = cov_collectors[QKV_NAMES["value"]].XXT

    # --q_centered: subtract the mean post-RoPE key outer product, so q_proj's row
    # metric is the key COVARIANCE rather than the second moment. The exact softmax
    # Jacobian is Cov_{p_t}(k) (diag(p) - p p^T, and sum_u p_tu = 1 makes any
    # constant cancel), so a constant offset in k -- overwhelmingly the k_proj bias
    # on Qwen -- is invisible to attention. BoA's uncentred metric spends its
    # budget on that direction. Kronecker structure is untouched.
    if q_centered and kmean is not None:
        _src = 'rot_out_K' if rotary_matrix is not None else QKV_NAMES["key"]
        # Do the subtraction in float64: centring cancels ~95% of the mass on Qwen,
        # and in float32 that leaves eigenvalues around -5e-4 of the max -- close
        # enough to BoA's per-head damping (~1% of the mean diagonal) that a
        # Cholesky could fail mid-run. Then project to PSD, which the exact
        # covariance is by construction; the clamp only removes round-off.
        _Y = cov_collectors[_src].YYT.double()
        _mu = kmean.mean().to(_Y.dtype)                                # [n_kv, d_h]
        _Y = _Y - 2.0 * (_mu[:, :, None] * _mu[:, None, :])
        _Y = 0.5 * (_Y + _Y.transpose(-1, -2))
        _w, _V = torch.linalg.eigh(_Y)
        _neg = (_w < 0).sum().item()
        _Y = _V @ torch.diag_embed(_w.clamp_min(0)) @ _V.transpose(-1, -2)
        if _neg:
            print(f"[q_centered] clamped {_neg} negative eigenvalue(s) "
                  f"(min {_w.min().item():.3e}, max {_w.max().item():.3e})")
        cov_collectors[_src].YYT = _Y.to(cov_collectors[_src].YYT.dtype)

    # Assign H_row for query/key
    if n_kv_heads != n_heads:
        n_shared = n_heads // n_kv_heads
        cov_collectors['rot_out_Q'].YYT = cov_collectors['rot_out_Q'].YYT.reshape(n_kv_heads, n_shared, head_dim, head_dim).mean(dim=1)
        cov_collectors['rot_out_K'].YYT = cov_collectors['rot_out_K'].YYT[:, None, :, :].expand(-1, n_shared, -1, -1).reshape(n_heads, head_dim, head_dim)
    
    if rotary_matrix is None:
        wrappers[QKV_NAMES["query"]].H_row = cov_collectors[QKV_NAMES["key"]].YYT
        wrappers[QKV_NAMES["key"]].H_row = cov_collectors[QKV_NAMES["query"]].YYT
    else:
        rotary_matrix = rotary_matrix.cuda()
        wrappers[QKV_NAMES["query"]].H_row = (rotary_matrix.transpose(-1, -2) @ cov_collectors['rot_out_K'].YYT @ rotary_matrix).mean(0)
        wrappers[QKV_NAMES["key"]].H_row = (rotary_matrix.transpose(-1, -2) @ cov_collectors['rot_out_Q'].YYT @ rotary_matrix).mean(0)

    # --q_identity: control arm. If BoA's q_proj row metric is mostly directions the
    # softmax cannot see, replacing it with I (which reduces the two-sided solve to
    # plain per-row GPTQ) should cost little.
    if q_identity:
        _q = QKV_NAMES["query"]
        _d = wrappers[_q].H_row.shape[-1]
        wrappers[_q].H_row = torch.eye(_d, device=wrappers[_q].H_row.device,
                                       dtype=wrappers[_q].H_row.dtype
                                       ).expand(n_heads, _d, _d).contiguous()

    # RSQ: overwrite the H_col of the chosen layers with the token-weighted one and,
    # for the one-sided baseline, drop the q/k row metric.
    if rsq is not None and rsq.get('rsq_weights'):
        for name, wrapper in wrappers.items():
            src = QKV_NAMES["value"] if name in [QKV_NAMES["query"], QKV_NAMES["key"]] else name
            if src in rsq_cov and not (name == QKV_NAMES["value"] and block_v):
                wrapper.H_col = rsq_cov[src][0]
            if rsq.get('rsq_col_only') and name in [QKV_NAMES["query"], QKV_NAMES["key"]]:
                wrapper.H_row = None

    # Phase 4: MLP-aware row metric for fc1 (ReLU-gated); off by default.
    if row_metric_fc1:
        W2 = layers['fc2'].weight.data.float()
        H_row_fc1 = (W2.T @ W2) * (coact.C / max(coact.n, 1))    # [d_ff, d_ff]
        # boa() eliminates rows sequentially per "head": a single 3072-row head costs
        # 3072 x 768 Python steps per layer (~20 min measured). Use a BLOCK-DIAGONAL
        # row metric with `groups` contiguous row groups (cross-group row coupling
        # dropped) so the groups vectorise like attention heads.
        g = row_metric_fc1 if isinstance(row_metric_fc1, int) and row_metric_fc1 > 1 else 48
        d_ff = H_row_fc1.shape[0]; r = d_ff // g
        wrappers['fc1'].H_row = torch.stack([H_row_fc1[i*r:(i+1)*r, i*r:(i+1)*r] for i in range(g)])  # [g, r, r]
        del coact

    # Assign H_row for value (paper eq. 9); off by default so the default path is
    # byte-identical to upstream.
    if row_metric_v:
        wrappers[QKV_NAMES['value']].H_row = value_row_metric(
            transformer_block, n_heads, n_kv_heads, head_dim,
            device=wrappers[QKV_NAMES['value']].H_col.device)

    del cov_collectors
    cleanup_memory(verbose=False)