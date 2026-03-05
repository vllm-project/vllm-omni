from typing import Any, Dict, List, Optional, Tuple

import torch

from nemo.collections.asr.modules import rnnt_abstract
from nemo.collections.asr.modules.conv_transformer_encoder import init_weights
from nemo.collections.asr.parts import rnnt_utils
from nemo.collections.common.parts import rnn
from nemo.collections.common.parts.mem_transformer import RelTransformerBlock
from nemo.collections.asr.models.configs import convtt_models_config as cfg
from nemo.core.classes import typecheck
from nemo.core.neural_types import (
    ElementType,
    EmbeddedTextType,
    LabelsType,
    LengthsType,
    NeuralType,
)


class TransformerTDecoder(rnnt_abstract.AbstractRNNTDecoder):
    @property
    def input_types(self):
        """Returns definitions of module input ports.
        """
        return {
            "targets": NeuralType(('B', 'T'), LabelsType()),
            "target_length": NeuralType(tuple('B'), LengthsType()),
            "states": NeuralType(('D', 'B', 'D'), ElementType(), optional=True),
        }

    @property
    def output_types(self):
        """Returns definitions of module output ports.
        """
        return {
            "outputs": NeuralType(('B', 'D', 'T'), EmbeddedTextType()),
            "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
        }

    def __init__(self,
                 vocab_size: int,
                 embed_size: int,
                 embed_dropout: float,
                 embed_proj_size: int,
                 sos_idx: Optional[int],
                 transformer_block: cfg.RelTransformerBlock,
                 blank_pos: str = 'vocab_last',
                 blank_as_pad: bool = True,
                 mask_label_prob: float = 0.0,
                 mask_label_id: Optional[int] = None,
                 norm_embed: bool = False,
                 norm_embed_proj: bool = False,
                 ln_eps: float = 1e-5,
                 init_mode='xavier_uniform', bias_init_mode='zero', embedding_init_mode='xavier_uniform'):
        if blank_pos == 'vocab_last':
            self.blank_idx = vocab_size - 1
        elif blank_pos == 'vocab_first':
            self.blank_idx = 0
        else:
            assert blank_pos == 'after_vocab_last'
            self.blank_idx = vocab_size

        super().__init__(vocab_size=vocab_size, blank_idx=self.blank_idx, blank_as_pad=blank_as_pad)

        self.transformer_block_cfg = transformer_block
        self.sos_idx = sos_idx
        self.prepend_sos_label = self.sos_idx is not None
        if self.prepend_sos_label:
            assert self.sos_idx >= 0

        if self.blank_as_pad:
            # if blank is used as pad, ensure blank is in the input embedding
            embed_num = vocab_size + 1 if blank_pos == 'after_vocab_last' else vocab_size
            padding_idx = self.blank_idx
        else:
            embed_num = vocab_size
            padding_idx = None
        self.embed = torch.nn.Embedding(embed_num, embed_size, padding_idx=padding_idx)

        self.embed_drop = torch.nn.Dropout(embed_dropout)

        if norm_embed:
            self.embed_norm = torch.nn.LayerNorm(embed_size, eps=ln_eps)
        else:
            self.embed_norm = identity

        if embed_proj_size > 0:
            self.embed_proj = torch.nn.Linear(embed_size, embed_proj_size)
        else:
            self.embed_proj = identity

        if norm_embed_proj:
            assert embed_proj_size > 0
            self.embed_proj_norm = torch.nn.LayerNorm(embed_proj_size, eps=ln_eps)
        else:
            self.embed_proj_norm = identity

        self.mask_label_prob = mask_label_prob
        self.mask_label_id = mask_label_id
        assert 0.0 <= self.mask_label_prob < 1.0
        if self.mask_label_prob > 0:
            assert self.mask_label_id is not None and self.mask_label_id >= 0
            assert self.mask_label_id not in [self.sos_idx, self.blank_idx]

        self.transformer_block = RelTransformerBlock(**self.transformer_block_cfg, ln_eps=ln_eps)

        self.apply(lambda x: init_weights(x, mode=init_mode, bias_mode=bias_init_mode,
                                          embedding_mode=embedding_init_mode))

    @typecheck()
    def forward(self, targets, target_length, states=None):
        # y: [B, U]
        y = rnn.label_collate(targets)

        is_decoding = states is not None
        if self.prepend_sos_label and not is_decoding:
            y = torch.nn.functional.pad(y, [1, 0], value=self.sos_idx)
            # we pad y, not targets, so target_length do not change
            # target_length = target_length + 1

        if self.mask_label_prob > 0 and self.training:
            y = random_replace(y, rep_prob=self.mask_label_prob, rep_id=self.mask_label_id)

        # state maintenance is unnecessary during training forward call
        # to get state, use .predict() method.
        h, _ = self.predict(y, state=states, add_sos=not self.prepend_sos_label)
        # [B, U, H] => [B, H, U+1]
        h = h.transpose(1, 2)

        return h, target_length

    def predict(self,
                y: Optional[torch.Tensor] = None,
                state: Optional[List[torch.Tensor]] = None,
                add_sos: bool = True,
                batch_size: Optional[int] = None) -> (torch.Tensor, List[torch.Tensor]):
        # Get device and dtype of current module
        _p = next(self.parameters())
        device = _p.device
        dtype = _p.dtype

        # If y is not None, it is of shape [B, U] with dtype long.
        if y is not None:
            if y.device != device:
                y = y.to(device)
            # (B, U) -> (B, U, H)
            h = self.embed(y)
            h = self.embed_drop(h)
            h = self.embed_norm(h)
        else:
            assert not self.prepend_sos_label
            # Y is not provided, assume state tensor is required
            # Emulates output of embedding of pad token.
            if batch_size is None:
                B = 1 if state is None else state[0][0].size(1)
            else:
                B = batch_size

            h = torch.zeros((B, 1, self.embed.embedding_dim), device=device, dtype=dtype)

        # Prepend blank "start of sequence" symbol (zero tensor)
        if add_sos:
            assert not self.prepend_sos_label
            B, U, H = h.shape
            start = torch.zeros((B, 1, H), device=h.device, dtype=h.dtype)
            h = torch.cat([start, h], dim=1).contiguous()  # (B, U + 1, H)
        else:
            start = None  # makes del call later easier

        h = self.embed_proj(h)
        h = self.embed_proj_norm(h)

        # [B, U+1, H] => [U+1, B, H]
        h = h.transpose(0, 1)
        h, new_state, _ = self.transformer_block(h, mems=state)
        # [U+1, B, H] => [B, U+1, H]
        h = h.transpose(0, 1)

        del start, state
        return h, new_state

    def initialize_state(self, batch_size, dtype, device) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        # assume y has shape [T, B, D]
        transformer_block_params = self.transformer_block_cfg
        n_layer = transformer_block_params['n_layer']
        n_head = transformer_block_params['n_head']
        d_head = transformer_block_params['d_head']
        use_mq_attn = transformer_block_params.get('use_mq_attn', False)

        if use_mq_attn:
            init_state = [(torch.zeros(0, batch_size, d_head, dtype=dtype, device=device),
                           torch.zeros(0, batch_size, d_head, dtype=dtype, device=device),
                           torch.tensor(0, dtype=torch.int32, device=device)) for _ in range(n_layer)]
        else:
            init_state = [(torch.zeros(0, batch_size, n_head, d_head, dtype=dtype, device=device),
                           torch.zeros(0, batch_size, n_head, d_head, dtype=dtype, device=device),
                           torch.tensor(0, dtype=torch.int32, device=device)) for _ in range(n_layer)]
        return init_state

    def score_hypothesis(
        self, hypothesis: rnnt_utils.Hypothesis, cache: Dict[Tuple[int], Any]
    ) -> (torch.Tensor, List[torch.Tensor], torch.Tensor):
        """
        Similar to the predict() method, instead this method scores a Hypothesis during beam search.
        Hypothesis is a dataclass representing one hypothesis in a Beam Search.

        Args:
            hypothesis: Refer to rnnt_utils.Hypothesis.
            cache: Dict which contains a cache to avoid duplicate computations.

        Returns:
            Returns a tuple (y, states, lm_token) such that:
            y is a torch.Tensor of shape [1, 1, H] representing the score of the last token in the Hypothesis.
            state is a list of RNN states, each of shape [L, 1, H].
            lm_token is the final integer token of the hypothesis.
        """
        if hypothesis.dec_state is not None:
            device = hypothesis.dec_state[0].device
        else:
            _p = next(self.parameters())
            device = _p.device

        # parse "blank" tokens in hypothesis
        if len(hypothesis.y_sequence) > 0 and hypothesis.y_sequence[-1] == self.blank_idx:
            blank_state = True
        else:
            blank_state = False

        # Convert last token of hypothesis to torch.Tensor
        target = torch.full([1, 1], fill_value=hypothesis.y_sequence[-1], device=device, dtype=torch.long)
        lm_token = target[:, -1]  # [1]

        # Convert current hypothesis into a tuple to preserve in cache
        sequence = tuple(hypothesis.y_sequence)

        if sequence in cache:
            y, new_state = cache[sequence]
        else:
            # Obtain score for target token and new states
            if blank_state:
                y, new_state = self.predict(None, state=None, add_sos=False, batch_size=1)  # [1, 1, H]

            else:
                y, new_state = self.predict(
                    target, state=hypothesis.dec_state, add_sos=False, batch_size=1
                )  # [1, 1, H]

            y = y[:, -1:, :]  # Extract just last state : [1, 1, H]
            cache[sequence] = (y, new_state)

        return y, new_state, lm_token

    def batch_score_hypothesis(
        self, hypotheses: List[rnnt_utils.Hypothesis], cache: Dict[Tuple[int], Any], batch_states: List[torch.Tensor]
    ) -> (torch.Tensor, List[torch.Tensor], torch.Tensor):
        """
        Used for batched beam search algorithms. Similar to score_hypothesis method.

        Args:
            hypothesis: List of Hypotheses. Refer to rnnt_utils.Hypothesis.
            cache: Dict which contains a cache to avoid duplicate computations.
            batch_states: List of torch.Tensor which represent the states of the RNN for this batch.
                Each state is of shape [L, B, H]

        Returns:
            Returns a tuple (b_y, b_states, lm_tokens) such that:
            b_y is a torch.Tensor of shape [B, 1, H] representing the scores of the last tokens in the Hypotheses.
            b_state is a list of list of RNN states, each of shape [L, B, H].
                Represented as B x List[states].
            lm_token is a list of the final integer tokens of the hypotheses in the batch.
        """
        final_batch = len(hypotheses)

        if final_batch == 0:
            raise ValueError("No hypotheses was provided for the batch!")

        _p = next(self.parameters())
        device = _p.device
        dtype = _p.dtype

        tokens = []
        process = []
        done = [None for _ in range(final_batch)]

        # For each hypothesis, cache the last token of the sequence and the current states
        for i, hyp in enumerate(hypotheses):
            sequence = tuple(hyp.y_sequence)

            if sequence in cache:
                done[i] = cache[sequence]
            else:
                tokens.append(hyp.y_sequence[-1])
                process.append((sequence, hyp.dec_state))

        if process:
            batch = len(process)

            # convert list of tokens to torch.Tensor, then reshape.
            tokens = torch.tensor(tokens, device=device, dtype=torch.long).view(batch, -1)
            dec_states = self.initialize_state(tokens.to(dtype=dtype))  # [L, B, H]
            dec_states = self.batch_initialize_states(dec_states, [d_state for seq, d_state in process])

            y, dec_states = self.predict(
                tokens, state=dec_states, add_sos=False, batch_size=batch
            )  # [B, 1, H], List([L, 1, H])

        # Update done states and cache shared by entire batch.
        j = 0
        for i in range(final_batch):
            if done[i] is None:
                # Select sample's state from the batch state list
                new_state = self.batch_select_state(dec_states, j)

                # Cache [1, H] scores of the current y_j, and its corresponding state
                done[i] = (y[j], new_state)
                cache[process[j][0]] = (y[j], new_state)

                j += 1

        # Set the incoming batch states with the new states obtained from `done`.
        batch_states = self.batch_initialize_states(batch_states, [d_state for y_j, d_state in done])

        # Create batch of all output scores
        # List[1, 1, H] -> [B, 1, H]
        batch_y = torch.stack([y_j for y_j, d_state in done])

        # Extract the last tokens from all hypotheses and convert to a tensor
        lm_tokens = torch.tensor([h.y_sequence[-1] for h in hypotheses], device=device, dtype=torch.long).view(
            final_batch
        )

        return batch_y, batch_states, lm_tokens

    def batch_initialize_states(self, batch_states: List[torch.Tensor], decoder_states: List[List[torch.Tensor]]):
        """
        Create batch of decoder states.

       Args:
           batch_states (list): batch of decoder states
              ([L x (B, H)], [L x (B, H)])

           decoder_states (list of list): list of decoder states
               [B x ([L x (1, H)], [L x (1, H)])]

       Returns:
           batch_states (tuple): batch of decoder states
               ([L x (B, H)], [L x (B, H)])
       """
        # LSTM has 2 states
        for layer in range(self.pred_rnn_layers):
            for state_id in range(len(batch_states)):
                batch_states[state_id][layer] = torch.stack([s[state_id][layer] for s in decoder_states])

        return batch_states

    def batch_select_state(self, batch_states: List[torch.Tensor], idx: int) -> List[List[torch.Tensor]]:
        """Get decoder state from batch of states, for given id.

        Args:
            batch_states (list): batch of decoder states
                ([L x (B, H)], [L x (B, H)])

            idx (int): index to extract state from batch of states

        Returns:
            (tuple): decoder states for given id
                ([L x (1, H)], [L x (1, H)])
        """
        state_list = []
        for state_id in range(len(batch_states)):
            states = [batch_states[state_id][layer][idx] for layer in range(self.pred_rnn_layers)]
            state_list.append(states)

        return state_list


def random_replace(inputs: torch.Tensor, rep_prob, rep_id):
    mask = torch.bernoulli(torch.full(inputs.size(), rep_prob, device=inputs.device)).type(inputs.dtype)
    return mask * rep_id + (1 - mask) * inputs


def identity(x):
    return x
