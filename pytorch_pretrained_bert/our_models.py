import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from modeling import BertPreTrainedModel, BertModel, BertSelfAttention

# Class (and class comments) based off of Huggingface's BertForQuestionAnswering example class
# todo: modify the class below to use method in the paper that we find (method on top of bert embeddings)

class BertWithAnswerVerifier(BertPreTrainedModel):
    """BERT model for Question Answering (span extraction).
    This module is composed of the BERT model with a linear layer on top of
    the sequence output that computes start_logits and end_logits

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `start_positions`: position of the first token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.
        `end_positions`: position of the last token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.

    Outputs:
        if `start_positions` and `end_positions` are not `None`:
            Outputs the total_loss which is the sum of the CrossEntropy loss for the start and end token positions.
        if `start_positions` or `end_positions` is `None`:
            Outputs a tuple of start_logits, end_logits which are the logits respectively for the start and end
            position tokens of shape [batch_size, sequence_length].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForQuestionAnswering(config)
    start_logits, end_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        #self.MAX_SEQ_LEN = 384 + 1 # note: we used max_seq_len of 384 for training, plus one for nonce
        super(BertWithAnswerVerifier, self).__init__(config)
        self.bert = BertModel(config)

        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        self.encoder = nn.lstm(input_size=config.hidden_size, hidden_size=config.hidden_size, num_layers=1, bidirectional=True)
        self.answer_verifier = nn.Linear(2 * config.hidden_size, 1)

        #self.answer_verifier = nn.Linear(config.hidden_size * self.MAX_SEQ_LEN, 1)
        # use cnn or lstm to predict whether answerable?
        # could also use attention, perhaps leveraging bert self attention class or other class from modeling.py
        # answerable_logit = self.answer_verifier(sequence_output.flatten())
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None, end_positions=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # todo: use packed padded sequence to represent seq_output_transposed
        seq_output_transposed = sequence_output.transpose(0, 1)  # (seq_len, batch, input_size)
        output, enc_state = self.encoder(seq_output_transposed)
        h_n, c_n = enc_state
        answerable_logit = self.answer_verifier(h_n)

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)

            combined_positions = start_positions + end_positions
            answerable_ground_truth = combined_positions.where(combined_positions == 0, 1, 0)
            answer_verifier_loss = loss_fct(answerable_logit, answerable_ground_truth)

            total_loss = (start_loss + end_loss) / 2 + answer_verifier_loss
            return total_loss
        else:
            return start_logits, end_logits