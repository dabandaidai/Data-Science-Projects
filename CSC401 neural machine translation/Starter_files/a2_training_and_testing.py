from tqdm import tqdm

import torch
import torch.nn as nn
import a2_bleu_score
import a2_dataloader
import a2_encoder_decoder

"""
This code is provided solely for the personal and private use of students
taking the CSC401H/2511H course at the University of Toronto. Copying for
purposes other than this use is expressly prohibited. All forms of
distribution of this code, including but not limited to public repositories on
GitHub, GitLab, Bitbucket, or any other online platform, whether as given or
with any changes, are expressly prohibited.

Authors: Sean Robertson, Jingcheng Niu, Zining Zhu, and Mohamed Abdall
Updated by: Raeid Saqur <raeidsaqur@cs.toronto.edu>

All of the files in this directory and all subdirectories are:
Copyright (c) 2023 University of Toronto
"""

'''Functions related to training and testing.

You don't need anything more than what's been imported here.
'''


def train_for_epoch(
        model: a2_encoder_decoder.EncoderDecoder,
        dataloader: a2_dataloader.HansardDataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device) -> float:
    """Train an EncoderDecoder for an epoch

    An epoch is one full loop through the training data. This function:

    1. Defines a loss function using :class:`torch.nn.CrossEntropyLoss`,
       keeping track of what id the loss considers "padding"
    2. For every iteration of the `dataloader` (which yields triples
       ``source_x, source_x_lens, target_y``)
       1. Sends ``source_x`` to the appropriate device via ``source_x = source_x.to(device)``. Same
          for ``source_x_lens`` and ``target_y``.
       2. Zeros out the model's previous gradient with ``optimizer.zero_grad()``
       3. Calls ``logits = model(source_x, source_x_lens, target_y)`` to determine next-token
          probabilities.
       4. Modifies ``target_y`` for the loss function, getting rid of a token and
          replacing excess end-of-sequence tokens with padding using
        ``model.get_target_padding_mask()`` and ``torch.masked_fill``
       5. Flattens out the sequence dimension into the batch dimension of both
          ``logits`` and ``target_y``
       6. Calls ``loss = loss_fn(logits, target_y)`` to calculate the batch loss
       7. Calls ``loss.backward()`` to backpropagate gradients through
          ``model``
       8. Calls ``optim.step()`` to update model parameters
    3. Returns the average loss over sequences

    Parameters
    ----------
    model : EncoderDecoder
        The model we're training.
    dataloader : HansardDataLoader
        Serves up batches of data.
    device : torch.device
        A torch device, like 'cpu' or 'cuda'. Where to perform computations.
    optimizer : torch.optim.Optimizer
        Implements some algorithm for updating parameters using gradient
        calculations.

    Returns
    -------
    avg_loss : float
        The total loss divided by the total numer of sequence
    """

    losses = 0
    num = 0

    # loss function
    func = nn.CrossEntropyLoss(ignore_index=model.source_pad_id)
    for source_x, source_x_lens, target_y in tqdm(dataloader):
        source_x = source_x.to(device)
        source_x_lens = source_x_lens.to(device)
        target_y = target_y.to(device)
        optimizer.zero_grad()
        logit = model(source_x, source_x_lens, target_y)
        # replace eos with mask
        excess = model.get_target_padding_mask(target_y)
        target_y = target_y.masked_fill(excess, model.source_pad_id)
        # flatten sequence dimension into batch dimension
        flattened = torch.flatten(logit, start_dim=0, end_dim=1)
        flat_E = torch.flatten(target_y[1:, :], start_dim=0)
        # batch loss, backprop and update
        loss = func(flattened, flat_E)
        loss.backward()
        optimizer.step()

        losses += loss.item()
        num += 1
        del source_x, source_x_lens, target_y, logit, loss

    return losses / num


def compute_batch_total_bleu(
        target_y_ref: torch.LongTensor,
        target_y_cand: torch.LongTensor,
        target_sos: int,
        target_eos: int) -> float:
    """Compute the total BLEU score over elements in a batch

    Parameters
    ----------
    target_y_ref : torch.LongTensor
        A batch of reference transcripts of shape ``(T, M)``, including
        start-of-sequence tags and right-padded with end-of-sequence tags.
    target_y_cand : torch.LongTensor
        A batch of candidate transcripts of shape ``(T', M)``, also including
        start-of-sequence and end-of-sequence tags.
    target_sos : int
        The ID of the start-of-sequence tag in the target vocabulary.
    target_eos : int
        The ID of the end-of-sequence tag in the target vocabulary.

    Returns
    -------
    total_bleu : float
        The sum total BLEU score for across all elements in the batch. Use
        n-gram precision 4.
    """
    # you can use target_y_ref.tolist() to convert the LongTensor to a python list
    # of numbers
    # get M
    M = target_y_ref.shape[1]
    total_bleu = 0
    # convert to python list
    reference = target_y_ref.transpose(0, 1).tolist()
    candidate = target_y_cand.transpose(0, 1).tolist()
    # iterate through all reference and candidates
    for i in range(M):
        reference_list = []
        # add if not start or end tag
        for j in reference[i]:
            if int(j) != target_eos and int(j) != target_sos:
                reference_list.append(j)
        candidate_list = []
        # add if not start or end tag
        for j in candidate[i]:
            if int(j) != target_eos and int(j) != target_sos:
                candidate_list.append(j)
        # calculate bleu score
        bleu_score = a2_bleu_score.BLEU_score(reference_list, candidate_list, 4)
        total_bleu += bleu_score

    return total_bleu


def compute_average_bleu_over_dataset(
        model: a2_encoder_decoder.EncoderDecoder,
        dataloader: a2_dataloader.HansardDataLoader,
        target_sos: int,
        target_eos: int,
        device: torch.device) -> float:
    """Determine the average BLEU score across sequences

    This function computes the average BLEU score across all sequences in
    a single loop through the `dataloader`.

    1. For every iteration of the `dataloader` (which yields triples
       ``source_x, source_x_lens, target_y_ref``):
       1. Sends ``source_x`` to the appropriate device via ``source_x = source_x.to(device)``. Same
          for ``source_x_lens``. No need for ``target_y_cand``, since it will always be
          compared on the CPU.
       2. Performs a beam search by calling ``b_1 = model(source_x, source_x_lens)``
       3. Extracts the top path per beam as ``target_y_cand = b_1[..., 0]``
       4. Computes the total BLEU score of the batch using
          :func:`compute_batch_total_bleu`
    2. Returns the average per-sequence BLEU score

    Parameters
    ----------
    model : EncoderDecoder
        The model we're testing.
    dataloader : HansardDataLoader
        Serves up batches of data.
    target_sos : int
        The ID of the start-of-sequence tag in the target vocabulary.
    target_eos : int
        The ID of the end-of-sequence tag in the target vocabulary.

    Returns
    -------
    avg_bleu : float
        The total BLEU score summed over all sequences divided by the number of
        sequences

    Args:
        device:
    """
    count = 0
    bleu_score = 0
    for source_x, source_x_lens, target_y_ref in tqdm(dataloader):
        source_x = source_x.to(device)
        source_x_lens = source_x_lens.to(device)
        b_1 = model(source_x, source_x_lens)
        target_y_cand = b_1[..., 0]
        bleu = compute_batch_total_bleu(target_y_ref, target_y_cand, target_sos, target_eos)
        bleu_score += bleu
        count += source_x_lens.shape[0]
    average = bleu_score / count
    return average
