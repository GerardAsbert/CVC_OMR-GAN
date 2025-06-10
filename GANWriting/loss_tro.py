import torch
import Levenshtein as Lev
from load_data import vocab_size, tokens, num_tokens, index2letter

def recon_criterion(predict, target):
    return torch.mean(torch.abs(predict - target))

class LabelSmoothing(torch.nn.Module):
    "Implement label smoothing."
    def __init__(self, size, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = torch.nn.KLDivLoss(reduction='sum')
        #self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        if torch.max(target) >= self.size or torch.min(target) < 0:
            raise ValueError(f"Target values out of range: {torch.min(target)} to {torch.max(target)}, expected within [0, {self.size - 1}]")

        # Create a smoothed distribution
        true_dist = x.detach().clone()  # Detach x from the computation graph
        true_dist.fill_(self.smoothing / (self.size - 1))  # Uniform smoothing for all classes except the correct one
        true_dist.scatter_(1, target.detach().unsqueeze(1), self.confidence)  # Set the correct class probability to confidence level

        self.true_dist = true_dist
        return self.criterion(x, true_dist)

log_softmax = torch.nn.LogSoftmax(dim=-1)
crit_KL = LabelSmoothing(vocab_size, 0.3)
crit_CE = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

def fine(label_list):
    if type(label_list) != type([]):
        return [label_list]
    else:
        return label_list

class CER():
    def __init__(self):
        self.ed = 0
        self.len = 0

    def add(self, pred, gt):
        pred_label = torch.topk(pred, 1, dim=-1)[1].squeeze(-1) # b,t,83->b,t,1->b,t
        pred_label = pred_label.cpu().numpy()
        batch_size = pred_label.shape[0]
        eds = list()
        lens = list()
        for i in range(batch_size):
            pred_text = pred_label[i].tolist()
            gt_text = gt[i].cpu().numpy().tolist()

            gt_text = fine(gt_text)
            pred_text = fine(pred_text)
            for j in range(num_tokens):
                gt_text = list(filter(lambda x: x!=j, gt_text))
                pred_text = list(filter(lambda x: x!=j, pred_text))
            gt_text = ''.join([index2letter[c-num_tokens] for c in gt_text])
            pred_text = ''.join([index2letter[c-num_tokens] for c in pred_text])
            ed_value = Lev.distance(pred_text, gt_text)
            eds.append(ed_value)
            lens.append(len(gt_text))
        self.ed += sum(eds)
        self.len += sum(lens)
        #print(f"Added {batch_size} items. Total len: {self.len}, Total ed: {self.ed}")

    def fin(self):
        if self.len == 0:
            print("Warning: self.len is 0, returning 0 to avoid division by zero")
            return 0
        return 100 * (self.ed / self.len)
    
def mean_and_covariance(features):
    mu = torch.mean(features, dim=0)
    cov = torch.cov(features)
    return mu, cov

def frechet_distance(mu_real, cov_real, mu_gen, cov_gen):
    # Compute the Fréchet distance
    diff = mu_real - mu_gen
    cov_mean = torch.sqrt(cov_real.mm(cov_gen))
    fid = diff.dot(diff) + torch.trace(cov_real + cov_gen - 2 * cov_mean)
    return fid

def check_nan_inf(tensor, name):
    if torch.isnan(tensor).any():
        print(f"{name} contains NaNs! AAAAAAAAAAAAAAAAAAAAAAAAa")
    if torch.isinf(tensor).any():
        print(f"{name} contains Infs! AAAAAAAAAAAAAAAAAAAAAAAAAAAA")

