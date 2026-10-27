"""Course-provided skeleton for Part B: BiLSTM slot filling.

Same fixed-interface-with-student-fill pattern as `crf.py`: the
constructor and `predict()` are fully implemented, wiring the model
together per the README's specified architecture (embeddings → BiLSTM →
per-token softmax). The one thing left to implement is `forward()` — the
actual embedding → BiLSTM → tag-space projection pass. Look for the
`# TODO: implement` block below.
"""

from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn


class BiLSTMModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        tagset_size: int,
        embedding_dim: int = 100,
        hidden_dim: int = 128,
        use_crf: bool = True,
    ) -> None:
        """Define the model's layers per the README's architecture:
        embeddings -> BiLSTM -> per-token softmax.

        Args:
            vocab_size: Number of distinct tokens in the input vocabulary.
            tagset_size: Number of distinct BIO tags in the output space.
            embedding_dim: Dimensionality of token embeddings.
            hidden_dim: Hidden size of the LSTM (per direction; BiLSTM
                output is `2 * hidden_dim`).
            use_crf: If True (default, and what the submitted checkpoint
                uses), train with the CRF NLL and decode with Viterbi. If
                False, the model is the plain embeddings -> BiLSTM -> softmax
                tagger (cross-entropy + argmax); used only for the Part C
                ablation curve. The transition parameters are created either
                way so the state_dict layout is identical.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=0.3,
            bidirectional=True,
            batch_first=True,
        )
        self.hidden2tag = nn.Linear(2 * hidden_dim, tagset_size)

        # Optional +5 bonus: handwritten CRF layer (forward algorithm + Viterbi)
        # on top of BiLSTM emissions. `predict()` calls `viterbi_decode()` first.
        self.use_crf = use_crf
        self.transitions = nn.Parameter(torch.empty(tagset_size, tagset_size))
        self.start_transitions = nn.Parameter(torch.empty(tagset_size))
        self.end_transitions = nn.Parameter(torch.empty(tagset_size))
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        self.vocab: Any = None

    def forward(
        self, sentences_batch: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Run embedding -> BiLSTM -> linear projection to tag space.

        This is the part you implement.

        Args:
            sentences_batch: `(batch_size, max_seq_len)` tensor of token ids,
                padded to the longest sequence in the batch.
            lengths: `(batch_size,)` tensor of each sequence's true (unpadded)
                length. Typically used with `nn.utils.rnn.pack_padded_sequence`
                / `pad_packed_sequence` so the LSTM ignores padding.

        Returns:
            `(batch_size, max_seq_len, tagset_size)` tensor of per-token,
            per-tag logits (unnormalized — softmax/cross-entropy is applied outside this method).
        """
        device = self.embedding.weight.device
        sentences_batch = sentences_batch.to(device)
        # pack_padded_sequence wants lengths on CPU
        lengths_cpu = lengths.cpu()

        embedded = self.dropout(self.embedding(sentences_batch))
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.lstm(packed)
        padded, _ = nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True, total_length=sentences_batch.size(1)
        )
        padded = self.dropout(padded)
        return self.hidden2tag(padded)

    def viterbi_decode(
        self, emissions: torch.Tensor, lengths: torch.Tensor
    ) -> list[list[str]]:
        """Hand-written Viterbi decode over BiLSTM emissions.

        STUDENT-IMPLEMENTED, OPTIONAL — backs the README's "+5 bonus: add a
        CRF decoding layer on top of BiLSTM (handwritten forward/Viterbi)."
        Not implemented by default. If you implement this, `predict()`
        picks it up automatically (it calls this first and only falls back
        to plain argmax decoding if this raises `NotImplementedError`) —
        no flag or declaration needed anywhere else. Must be a genuinely
        hand-written forward/Viterbi pass over learned tag-transition
        scores; don't call out to an external CRF library here (e.g.
        `sklearn-crfsuite`) — that's Part A's job.

        Args:
            emissions: `(batch_size, max_seq_len, tagset_size)` tensor of
                per-token, per-tag logits — the same tensor `forward()`
                returns.
            lengths: `(batch_size,)` tensor of each sequence's true
                (unpadded) length.

        Returns:
            A list of predicted BIO label sequences (as tag strings), one
            per input sequence, aligned token-for-token and truncated to
            each sequence's true length.
        """
        if not self.use_crf:
            # Softmax-only ablation variant: let predict() fall back to argmax.
            raise NotImplementedError("use_crf=False: argmax decoding instead.")

        id_to_tag = self._id_to_tag()
        allowed = self._bio_masks(id_to_tag, emissions.size(-1), emissions.device)
        paths: list[list[str]] = []
        for batch_idx, length in enumerate(lengths.tolist()):
            length = int(length)
            path_ids = self._viterbi_one(emissions[batch_idx, :length], allowed)
            if id_to_tag is None:
                paths.append([str(tag_id) for tag_id in path_ids])
            else:
                paths.append([id_to_tag[tag_id] for tag_id in path_ids])
        return paths

    def _id_to_tag(self) -> dict[int, str] | None:
        """Tag names from `self.vocab`, accepting the common shapes it comes in:
        an object or a dict, with `id_to_tag` (dict with int or str keys, or a
        list) or only `tag_to_id`."""
        vocab = self.vocab
        if vocab is None:
            return None

        def field(name: str) -> Any:
            if isinstance(vocab, dict):
                return vocab.get(name)
            return getattr(vocab, name, None)

        mapping = field("id_to_tag")
        if mapping is None:
            tag_to_id = field("tag_to_id")
            if tag_to_id is not None:
                mapping = {int(i): t for t, i in tag_to_id.items()}
        if mapping is None:
            return None
        if isinstance(mapping, (list, tuple)):
            return dict(enumerate(mapping))
        return {int(k): v for k, v in mapping.items()}

    @staticmethod
    def _bio_masks(
        id_to_tag: dict[int, str] | None, num_tags: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Hard IOB2 constraints as additive masks (0 = allowed, -inf = illegal).

        start[j]  : a sequence cannot open with I-X.
        trans[i,j]: I-X may only follow B-X or I-X.
        Applied inside Viterbi, so the decoder never emits an illegal sequence
        whatever the learned transition scores are.
        """
        if id_to_tag is None:
            return None
        neg = float("-inf")
        start = torch.zeros(num_tags, device=device)
        trans = torch.zeros(num_tags, num_tags, device=device)
        for j in range(num_tags):
            to_tag = id_to_tag.get(j, "O")
            if not to_tag.startswith("I-"):
                continue
            start[j] = neg
            typ = to_tag[2:]
            for i in range(num_tags):
                if id_to_tag.get(i, "O") not in (f"B-{typ}", f"I-{typ}"):
                    trans[i, j] = neg
        return start, trans

    def _viterbi_one(
        self,
        emissions: torch.Tensor,
        allowed: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> list[int]:
        """Viterbi decode a single sequence of emission logits `(L, num_tags)`.

        `allowed` = (start_mask, transition_mask) from `_bio_masks()`, added to
        the learned start/transition scores so illegal IOB2 paths score -inf.
        """
        seq_len, num_tags = emissions.shape
        trellis = torch.empty_like(emissions)
        backpointers = torch.zeros(
            (seq_len, num_tags), dtype=torch.long, device=emissions.device
        )

        start = self.start_transitions
        transitions = self.transitions
        if allowed is not None:
            start = start + allowed[0]
            transitions = transitions + allowed[1]

        trellis[0] = start + emissions[0]
        for t in range(1, seq_len):
            scores = trellis[t - 1].unsqueeze(1) + transitions
            max_score, bp = scores.max(dim=0)
            trellis[t] = max_score + emissions[t]
            backpointers[t] = bp

        trellis[-1] = trellis[-1] + self.end_transitions
        best_tag = int(trellis[-1].argmax().item())

        path = [best_tag]
        for t in range(seq_len - 1, 0, -1):
            best_tag = int(backpointers[t, best_tag].item())
            path.append(best_tag)
        path.reverse()
        return path

    def crf_nll(
        self, emissions: torch.Tensor, tags: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Mean CRF negative log-likelihood of gold `tags` given `emissions`."""
        gold = self._gold_path_score(emissions, tags, lengths)
        log_z = self._log_partition(emissions, lengths)
        return (log_z - gold).mean()

    def _gold_path_score(
        self, emissions: torch.Tensor, tags: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Score of the gold tag sequence: start + emissions + transitions + end."""
        batch_size = emissions.size(0)
        max_len = emissions.size(1)
        device = emissions.device
        tags = tags.to(device)
        lengths = lengths.to(device)
        batch_ix = torch.arange(batch_size, device=device)

        score = self.start_transitions[tags[:, 0]] + emissions[batch_ix, 0, tags[:, 0]]
        for t in range(1, max_len):
            mask = (t < lengths).float()
            trans = self.transitions[tags[:, t - 1], tags[:, t]]
            emit = emissions[batch_ix, t, tags[:, t]]
            score = score + mask * (trans + emit)

        last_tags = tags[batch_ix, lengths - 1]
        return score + self.end_transitions[last_tags]

    def _log_partition(
        self, emissions: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Log-partition function (forward algorithm) for each sequence in the batch."""
        batch_size, max_len, _ = emissions.shape
        lengths = lengths.to(emissions.device)

        alpha = self.start_transitions + emissions[:, 0]
        for t in range(1, max_len):
            next_alpha = torch.logsumexp(
                alpha.unsqueeze(2)
                + self.transitions.unsqueeze(0)
                + emissions[:, t].unsqueeze(1),
                dim=1,
            )
            mask = (t < lengths).unsqueeze(1)
            alpha = torch.where(mask, next_alpha, alpha)

        return torch.logsumexp(alpha + self.end_transitions, dim=1)

    def apply_iob_constraints(self, id_to_tag: dict[int, str]) -> None:
        """Hard-forbid illegal IOB2 transitions (I-X not after B-X / I-X)."""
        with torch.no_grad():
            for i, from_tag in id_to_tag.items():
                if from_tag.startswith("I-"):
                    self.start_transitions[i] = -1e4
                for j, to_tag in id_to_tag.items():
                    if not to_tag.startswith("I-"):
                        continue
                    typ = to_tag[2:]
                    if from_tag not in (f"B-{typ}", f"I-{typ}"):
                        self.transitions[i, j] = -1e4

    def predict(self, sentences: list[list[str]], vocab: Any) -> list[list[str]]:
        """Predict BIO label sequences for tokenized sentences.

        Gives `BiLSTMModel` the same `.predict()` signature as `CRFModel`,
        so `autograder_interface.py` can call either model uniformly.

        Args:
            sentences: A list of tokenized sentences (raw token strings).
            vocab: Vocabulary/label object exposing the token<->id and
                tag<->id mappings built by `data_loader.py` (e.g.
                `vocab.token_to_id`, `vocab.id_to_tag`). Adjust the exact
                attribute names below to match `data_loader.py`'s actual
                interface once it exists.

        Returns:
            A list of predicted BIO label sequences, one per sentence,
            aligned token-for-token with `sentences`.
        """
        # Viterbi bonus reads tag names off `self.vocab`; keep it in sync with
        # the vocab object the caller passed (load_model already sets this).
        if vocab is not None:
            self.vocab = vocab

        # --- Integration point: `data_loader.py`'s vocab interface. ---------
        unk_id = vocab.token_to_id.get("<UNK>", 0)
        id_sequences = [
            [vocab.token_to_id.get(token, unk_id) for token in sentence]
            for sentence in sentences
        ]
        # ---------------------------------------------------------------------

        device = next(self.parameters()).device
        lengths = torch.tensor(
            [len(seq) for seq in id_sequences], dtype=torch.long, device=device
        )
        max_len = int(lengths.max().item())

        padded = torch.zeros((len(id_sequences), max_len), dtype=torch.long, device=device)
        for idx, seq in enumerate(id_sequences):
            padded[idx, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)

        self.eval()
        with torch.no_grad():
            logits = self.forward(padded, lengths)

            try:
                return self.viterbi_decode(logits, lengths)
            except NotImplementedError:
                pass

            predicted_ids = logits.argmax(dim=-1)

        predictions: list[list[str]] = []

        for idx, length in enumerate(lengths.tolist()):
            tag_ids = predicted_ids[idx, :length].tolist()
            predictions.append([vocab.id_to_tag[tag_id] for tag_id in tag_ids])

        return predictions
