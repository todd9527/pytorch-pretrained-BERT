from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import json
import random

import torch

from pytorch_pretrained_bert.our_models import BertWithAnswerVerifier

# todo: write some unit tests
def _test_combine_logits():
    start_logits = torch.zeros(3, 4)
    end_logits = torch.zeros(3, 4)
    answerability_logits = torch.ones(3, 4, 2)
    start_logits, end_logits = BertWithAnswerVerifier._combine_logits_with_verifier(start_logits, end_logits, answerability_logits)
    assert start_logits[0][0] == 1
    assert start_logits[0][1] == 0

_test_combine_logits()