import torch
from torch import autograd, nn
import torch.nn.functional as F

from .criterion import BaseCriterion, EqualizedConv1d  # , # PredictionNetwork


class _SOFT_ALIGN(autograd.Function):
    """Soft-align a set of predictions to some vectors.

    Args:
    - log_probs: BS x Num_X x Num_Preds giving log(X|P) (that is the probaility of emission and not symbol classification)

    Retursn:
    - costs (BS), alignments (BS x Num_X) if int's denoting which symbol best fits the value.

    """
    @staticmethod
    def _alignment_cost(log_probs, allowed_skips_beg, allowed_skips_end, force_forbid_blank):
        # log_probs is BS x WIN_LEN x NUM_PREDS
        bs, win_len, num_preds = log_probs.size()
        assert win_len >=  num_preds
        padded_log_probs = F.pad(
            log_probs, (0, 0, allowed_skips_beg, allowed_skips_end), "constant", 0)
        padded_win_len = win_len + allowed_skips_beg + allowed_skips_end
        fake_ctc_labels = torch.arange(1, num_preds+1, dtype=torch.int).expand(bs, num_preds)

        # append impossible BLANK probabilities
        ctc_log_probs = padded_log_probs.permute(1, 0, 2).contiguous()
        if force_forbid_blank:
            ctc_log_probs = torch.cat((
                torch.empty(padded_win_len, bs, 1, device=log_probs.device).fill_(-1000),
                ctc_log_probs
            ), 2)
        # Now ctc_log_probs is win_size x BS x (num_preds + 1)
        assert ctc_log_probs.is_contiguous()

        # normalize the log-probs over num_preds
        # This is required, because ctc returns a bad gradient when given 
        # unnormalized log probs
        log_sum_exps = torch.logsumexp(ctc_log_probs, 2, keepdim=True)
        ctc_log_probs = ctc_log_probs - log_sum_exps
        losses = F.ctc_loss(
            ctc_log_probs, 
            fake_ctc_labels,
            torch.empty(bs, dtype=torch.int).fill_(padded_win_len),
            torch.empty(bs, dtype=torch.int).fill_(num_preds),
            reduction='none')
        losses = losses - log_sum_exps.squeeze(2).sum(0)
        return losses

    @staticmethod
    def forward(ctx, log_probs, allowed_skips_beg=0, allowed_skips_end=0, force_forbid_blank=True):
        log_probs = log_probs.detach().requires_grad_()
        with torch.enable_grad():
            losses = _SOFT_ALIGN._alignment_cost(
                log_probs, allowed_skips_beg, allowed_skips_end, force_forbid_blank)
            losses.sum().backward()
            grads = log_probs.grad.detach()
        _, alignment = grads.min(-1)
        ctx.save_for_backward(grads)

        return losses.detach(), alignment

    @staticmethod
    def backward(ctx, grad_output, _):
        grads, = ctx.saved_tensors
        grad_output = grad_output.to(grads.device)
        return grads * grad_output.view(-1, 1, 1), None, None, None

soft_align = _SOFT_ALIGN.apply


class PredictionNetwork(nn.Module):

    def __init__(self,
                 nPredicts,
                 dimOutputAR,
                 dimOutputEncoder,
                 rnnMode=None,
                 dropout=False,
                 sizeInputSeq=116):

        super(PredictionNetwork, self).__init__()
        self.predictors = nn.ModuleList()
        self.RESIDUAL_STD = 0.01
        self.dimOutputAR = dimOutputAR

        self.dropout = nn.Dropout(p=0.5) if dropout else None
        for i in range(nPredicts):
            if rnnMode == 'RNN':
                self.predictors.append(
                    nn.RNN(dimOutputAR, dimOutputEncoder))
                self.predictors[-1].flatten_parameters()
            elif rnnMode == 'LSTM':
                self.predictors.append(
                    nn.LSTM(dimOutputAR, dimOutputEncoder, batch_first=True))
                self.predictors[-1].flatten_parameters()
            elif rnnMode == 'ffd':
                self.predictors.append(
                    FFNetwork(dimOutputAR, dimOutputEncoder,
                              dimOutputEncoder, 0))
            elif rnnMode == 'conv4':
                self.predictors.append(
                    ShiftedConv(dimOutputAR, dimOutputEncoder, 4))
            elif rnnMode == 'conv8':
                self.predictors.append(
                    ShiftedConv(dimOutputAR, dimOutputEncoder, 8))
            elif rnnMode == 'conv12':
                self.predictors.append(
                    ShiftedConv(dimOutputAR, dimOutputEncoder, 12))
            elif rnnMode == 'transformer':
                from transformers import buildTransformerAR
                self.predictors.append(
                    buildTransformerAR(dimOutputEncoder,
                                       1,
                                       sizeInputSeq,
                                       False))
            else:
                self.predictors.append(
                    nn.Linear(dimOutputAR, dimOutputEncoder, bias=False))
                if dimOutputEncoder > dimOutputAR:
                    residual = dimOutputEncoder - dimOutputAR
                    self.predictors[-1].weight.data.copy_(torch.cat([torch.randn(
                        dimOutputAR, dimOutputAR), self.RESIDUAL_STD * torch.randn(residual, dimOutputAR)], dim=0))

    def forward(self, c):

        out = []

        # UGLY
        if isinstance(self.predictors[0], EqualizedConv1d):
            c = c.permute(0, 2, 1)

        for k in range(len(self.predictors)):

            locC = self.predictors[k](c)
            if isinstance(locC, tuple):
                locC = locC[0]
            if isinstance(self.predictors[k], EqualizedConv1d):
                locC = locC.permute(0, 2, 1)
            if self.dropout is not None:
                locC = self.dropout(locC)
            locC = locC.view(locC.size(0), locC.size(1), locC.size(2), 1)
            out.append(locC)
        return torch.cat(out, 3)




class CPCUnsupersivedCriterion(BaseCriterion):

    def __init__(self,
                 nPredicts,             # Number of predictions
                 nMatched,                  # Window size to which align predictions
                 dimOutputAR,           # Dimension of G_ar
                 dimOutputEncoder,      # Dimension of the convolutional net
                 negativeSamplingExt,   # Number of negative samples to draw
                 allowed_skips_beg=0,     # number of predictions that we can skip at the beginning
                 allowed_skips_end=0,     # number of predictions that we can skip at the end
                 predict_self_loop=False, # always predict a repetition of the first symbol
                 no_negs_in_match_window=False,  # prevent sampling negatives from the matching window
                 learn_blank=False,       # try to use the blank symbol
                 normalize_enc=False,
                 normalize_preds=False,
                 masq_rules="",
                 loss_temp=1.0,
                 limit_negs_in_batch=None,
                 mode=None,
                 rnnMode=False,
                 dropout=False,
                 speakerEmbedding=0,
                 nSpeakers=0,
                 sizeInputSeq=128):

        print ("!!!!!!!!!USING CPCCTC!!!!!!!!!!!!")

        super(CPCUnsupersivedCriterion, self).__init__()
        if speakerEmbedding > 0:
            print(
                f"Using {speakerEmbedding} speaker embeddings for {nSpeakers} speakers")
            self.speakerEmb = torch.nn.Embedding(nSpeakers, speakerEmbedding)
            dimOutputAR += speakerEmbedding
        else:
            self.speakerEmb = None

        self.normalize_enc = normalize_enc
        self.normalize_preds = normalize_preds
        self.loss_temp = loss_temp
        self.nMatched = nMatched
        self.no_negs_in_match_window = no_negs_in_match_window
        self.wPrediction = PredictionNetwork(
            nPredicts, dimOutputAR, dimOutputEncoder, rnnMode=rnnMode,
            dropout=dropout, sizeInputSeq=sizeInputSeq - nMatched)
        self.learn_blank = learn_blank
        if learn_blank:
            self.blank_proto = torch.nn.Parameter(torch.zeros(1, 1, dimOutputEncoder, 1))
        else:
            self.register_parameter('blank_proto', None)
        self.nPredicts = nPredicts
        self.negativeSamplingExt = negativeSamplingExt
        self.allowed_skips_beg = allowed_skips_beg
        self.allowed_skips_end = allowed_skips_end
        self.predict_self_loop = predict_self_loop
        # if predict_self_loop:
        #     self.self_loop_gain = torch.nn.Parameter(torch.ones(1))
        # else:
        #     self.register_parameter('self_loop_gain', None)
        self.limit_negs_in_batch = limit_negs_in_batch

        if masq_rules:
            masq_buffer = torch.zeros(self.nMatched, self.nPredicts)
            for rule in masq_rules.split(','):
                a,b,c,d = [int(a) if a.lower() != "none" else None for a in rule.split(':')]
                masq_buffer[a:b,c:d] = 1
            print("!!!MasqBuffer: ", masq_buffer)
            self.register_buffer("masq_buffer", masq_buffer.unsqueeze(0))
        else:
            self.register_buffer("masq_buffer", None)

        if mode not in [None, "reverse"]:
            raise ValueError("Invalid mode")

        self.mode = mode

    def sampleClean(self, encodedData, windowSize):

        batchSize, nNegativeExt, dimEncoded = encodedData.size()
        outputs = []

        negExt = encodedData.contiguous().view(-1, dimEncoded)
        # Draw nNegativeExt * batchSize negative samples anywhere in the batch
        batchIdx = torch.randint(low=0, high=batchSize,
                                 size=(batchSize, 
                                       self.negativeSamplingExt * windowSize, ),
                                 device=encodedData.device)
        if self.limit_negs_in_batch:
            # sample nagatives from a small set of entries in minibatch
            batchIdx = torch.remainder(batchIdx, self.limit_negs_in_batch)
            batchBaseIdx = torch.arange(0, batchSize, device=encodedData.device)
            batchBaseIdx -= torch.remainder(batchBaseIdx, self.limit_negs_in_batch)
            batchIdx += batchBaseIdx.unsqueeze(1) 
            # we can get too large, if batchsize is not divisible by limit_negs_in_batch
            batchIdx = torch.remainder(batchIdx, batchSize)

            # if not  ((batchIdx.max().item() < batchSize) and 
            #          (batchIdx.min().item() >= 0)):
            #     import pdb; pdb.set_trace()
        batchIdx = batchIdx.contiguous().view(-1)

        if self.no_negs_in_match_window:
            idx_low = self.nMatched  # forbid sampling negatives in the prediction window
        else:
            idx_low = 1  # just forbid sampling own index for negative
        seqIdx = torch.randint(low=idx_low, high=nNegativeExt,
                               size=(self.negativeSamplingExt
                                     * windowSize * batchSize, ),
                               device=encodedData.device)

        baseIdx = torch.arange(0, windowSize, device=encodedData.device)
        baseIdx = baseIdx.view(1, 1,
                               windowSize).expand(1,
                                                  self.negativeSamplingExt,
                                                  windowSize).expand(batchSize, self.negativeSamplingExt, windowSize)
        seqIdx += baseIdx.contiguous().view(-1)
        seqIdx = torch.remainder(seqIdx, nNegativeExt)

        extIdx = seqIdx + batchIdx * nNegativeExt
        negExt = negExt[extIdx].view(batchSize, self.negativeSamplingExt,
                                     windowSize, dimEncoded)
        
        return negExt
        
        # labelLoss = torch.zeros((batchSize * windowSize),
        #                         dtype=torch.long,
        #                         device=encodedData.device)

        # for k in range(1, self.nMatched + 1):

        #     # Positive samples
        #     if k < self.nMatched:
        #         posSeq = encodedData[:, k:-(self.nMatched-k)]
        #     else:
        #         posSeq = encodedData[:, k:]

        #     posSeq = posSeq.view(batchSize, 1, posSeq.size(1), dimEncoded)
        #     fullSeq = torch.cat((posSeq, negExt), dim=1)
        #     outputs.append(fullSeq)

        # return outputs, labelLoss

    def forward(self, cFeature, encodedData, label, captureOptions=None):


        if self.mode == "reverse":
            encodedData = torch.flip(encodedData, [1])
            cFeature = torch.flip(cFeature, [1])

        batchSize, seqSize, dimAR = cFeature.size()
        windowSize = seqSize - self.nMatched

        cFeature = cFeature[:, :windowSize]

        if self.normalize_enc:
            encodedData = F.layer_norm(encodedData, (encodedData.size(-1),))

        # sampledData, labelLoss = self.sampleClean(encodedData, windowSize)
        # negatives: BS x Len x NumNegs x D
        sampledNegs = self.sampleClean(encodedData, windowSize).permute(0, 2, 1, 3)

        if self.speakerEmb is not None:
            l_ = label.view(batchSize, 1).expand(batchSize, windowSize)
            embeddedSpeaker = self.speakerEmb(l_)
            cFeature = torch.cat([cFeature, embeddedSpeaker], dim=2)

        # Predictions, BS x Len x D x nPreds
        predictions = self.wPrediction(cFeature)
        nPredicts = self.nPredicts

        extra_preds = []

        if self.learn_blank:
            extra_preds.append(self.blank_proto.expand(batchSize, windowSize, self.blank_proto.size(2), 1))

        if self.predict_self_loop:
            # old and buggy
            # extra_preds.append(cFeature.unsqueeze(-1))
            # new and shiny
            extra_preds.append(encodedData[:, :windowSize, :].unsqueeze(-1) )  # * self.self_loop_gain)

        if extra_preds:
            nPredicts += len(extra_preds)
            extra_preds.append(predictions)
            predictions = torch.cat(
                extra_preds, -1
            )

        if self.normalize_preds:
            predictions = F.layer_norm(predictions, (predictions.size(-1),))
        
        #predictions = torch.cat(predictions, 1).permute(0, 2, 3, 1)

        # predictions = self.wPrediction(cFeature)
        # predictions = torch.cat(predictions, 1)

        # Positive examples in the window, BS x Len x W x D
        positives = encodedData[:,1:].unfold(1, self.nMatched, 1).permute(0,1,3,2)
        # gt_and_neg = torch.cat((pred_windows, sampledData.permute(0, 2, 3, 1)), 3)

        # BS x L x NumNegs x NumPreds
        neg_log_scores = sampledNegs @ predictions / sampledNegs.size(-1)

        # BS x L x W x NumPreds
        pos_log_scores = positives @ predictions / sampledNegs.size(-1)

        # We now want ot get a matrix BS x L x W x NumPreds
        # in which each entry is the log-softmax of predicting a window elem in contrast to al negs

        # log(e^x_p / (e^x_p + \sum_n e^x_n))
        # first compute \log \sum_n e^x_n
        neg_log_tot_scores = torch.logsumexp(neg_log_scores, 2, keepdim=True)

        # now log(e^xp / (e^x_p + e^x_n)) 
        # this can be further optimized.
        log_scores = torch.log_softmax(
            torch.stack((pos_log_scores,
                         neg_log_tot_scores.expand_as(pos_log_scores)), 0), 
            dim=0)[0]
        
        log_scores = log_scores.view(batchSize*windowSize, self.nMatched, nPredicts)
        # print('ls-stats', log_scores.mean().item(), log_scores.std().item())
        if self.masq_buffer is not None:
            masq_buffer = self.masq_buffer
            if extra_preds:
                masq_buffer = torch.cat([masq_buffer[:, :, :1]] * (len(extra_preds) - 1) + [masq_buffer], dim=2)
            log_scores = log_scores.masked_fill(masq_buffer > 0, -1000)
        losses, aligns = soft_align(log_scores / self.loss_temp, self.allowed_skips_beg, self.allowed_skips_end, not self.learn_blank)
        losses = losses * self.loss_temp

        pos_is_selected = (pos_log_scores > neg_log_scores.max(2, keepdim=True)[0]).view(batchSize*windowSize, self.nMatched, nPredicts)

        # This is approximate Viterbi alignment loss and accurracy
        outLosses = -torch.gather(log_scores, 2, aligns.unsqueeze(-1)).squeeze(-1).float().mean(0, keepdim=True)
        outAcc = torch.gather(pos_is_selected, 2, aligns.unsqueeze(-1)).squeeze(-1).float().mean(0, keepdim=True)

        # just simulate a per-prediction loss
        outLossesD = outLosses.detach()
        losses = losses.mean() / outLossesD.sum() * outLossesD

        captureRes = None
        if captureOptions != None:
            for o in captureOptions:
                assert o in ('pred', 'cpcctc_align', 'cpcctc_log_scores', 'locals')
            captureRes = {}
            if 'pred' in captureOptions:
                # 1st sting in last dim can be self loop - need to keep as it's also being aligned
                captureRes['pred'] = predictions
            if 'cpcctc_align' in captureOptions:
                readableAligns = aligns.detach().view(batchSize, windowSize, self.nMatched)
                captureRes['cpcctc_align'] = readableAligns
            if 'cpcctc_log_scores' in captureOptions:
                captureRes['cpcctc_log_scores'] = log_scores.detach().view(batchSize, windowSize, self.nMatched, -1)
            if 'locals' in captureOptions:
                captureRes['locals'] = locals()

        return losses, outAcc, captureRes



