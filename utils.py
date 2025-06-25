from copy import deepcopy
from scipy.optimize import minimize

import torch
import torch.nn.functional as F
import numpy as np
from numpy import linalg as LA
import math
import copy

"""
Define task metrics, loss functions and model trainer here.
"""


class ConfMatrix(object):
    """
    For mIoU and other pixel-level classification tasks.
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def reset(self):
        self.mat = None

    def update(self, pred, target):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=pred.device)
        with torch.no_grad():
            k = (target >= 0) & (target < n)
            inds = n * target[k].to(torch.int64) + pred[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def get_metrics(self):
        h = self.mat.float()
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return torch.mean(iu).item()
    

def needs_mask(name):
    return name.endswith('weight')


def create_task_flags(task, dataset, with_noise=False):
    """
    Record task and its prediction dimension.
    Noise prediction is only applied in auxiliary learning.
    """
    nyu_tasks = {'seg': 13, 'depth': 1, 'normal': 3}
    cityscapes_tasks = {'seg': 19, 'part_seg': 10, 'disp': 1}

    tasks = {}
    if task != 'all':
        if dataset == 'nyuv2':
            tasks[task] = nyu_tasks[task]
        elif dataset == 'cityscapes':
            tasks[task] = cityscapes_tasks[task]
    else:
        if dataset == 'nyuv2':
            tasks = nyu_tasks
        elif dataset == 'cityscapes':
            tasks = cityscapes_tasks

    if with_noise:
        tasks['noise'] = 1
    return tasks


def get_weight_str(weight, tasks):
    """
    Record task weighting.
    """
    weight_str = 'Task Weighting | '
    for i, task_id in enumerate(tasks):
        weight_str += '{} {:.04f} '.format(task_id.title(), weight[i])
    return weight_str


def get_weight_str_ranked(weight, tasks, rank_num):
    """
    Record top-k ranked task weighting.
    """
    rank_idx = np.argsort(weight)

    if type(tasks) == dict:
        tasks = list(tasks.keys())

    top_str = 'Top {}: '.format(rank_num)
    bot_str = 'Bottom {}: '.format(rank_num)
    for i in range(rank_num):
        top_str += '{} {:.02f} '.format(tasks[rank_idx[-i-1]].title(), weight[rank_idx[-i-1]])
        bot_str += '{} {:.02f} '.format(tasks[rank_idx[i]].title(), weight[rank_idx[i]])

    return 'Task Weighting | {}| {}'.format(top_str, bot_str)

def get_acc_str_ranked(acc, tasks, rank_num):
    """
    Record top-k ranked task accuracy.
    """
    rank_idx = np.argsort(acc)

    if type(tasks) == dict:
        tasks = list(tasks.keys())

    top_str = 'Top {}: '.format(rank_num)
    bot_str = 'Bottom {}: '.format(rank_num)
    for i in range(rank_num):
        top_str += '{} {:.02f} '.format(tasks[rank_idx[-i-1]].title(), acc[rank_idx[-i-1]])
        bot_str += '{} {:.02f} '.format(tasks[rank_idx[i]].title(), acc[rank_idx[i]])

    return 'Task Performance | {}| {}'.format(top_str, bot_str)


def compute_loss(pred, gt, task_id):
    """
    Compute task-specific loss.
    """
    if task_id in ['seg', 'part_seg'] or 'class' in task_id:
        # Cross Entropy Loss with Ignored Index (values are -1)
        loss = F.cross_entropy(pred, gt, ignore_index=-1)

    if task_id in ['normal', 'depth', 'disp', 'noise']:
        # L1 Loss with Ignored Region (values are 0 or -1)
        invalid_idx = -1 if task_id == 'disp' else 0
        valid_mask = (torch.sum(gt, dim=1, keepdim=True) != invalid_idx).to(pred.device)
        loss = torch.sum(F.l1_loss(pred, gt, reduction='none').masked_select(valid_mask)) \
                / torch.nonzero(valid_mask, as_tuple=False).size(0)
    return loss


class TaskMetric:
    def __init__(self, train_tasks, pri_tasks, batch_size, epochs, dataset, include_mtl=False):
        self.train_tasks = train_tasks
        self.pri_tasks = pri_tasks
        self.batch_size = batch_size
        self.dataset = dataset
        self.include_mtl = include_mtl
        self.metric = {key: np.zeros([epochs, 2]) for key in train_tasks.keys()}
        self.data_counter = 0
        self.epoch_counter = 0
        self.conf_mtx = {}

        if include_mtl:
            self.metric['all'] = np.zeros(epochs)
        for task in self.train_tasks:
            if task in ['seg', 'part_seg']:
                self.conf_mtx[task] = ConfMatrix(self.train_tasks[task])

    def reset(self):
        """
        Reset data counter and confusion matrices.
        """
        self.epoch_counter += 1
        self.data_counter = 0

        if len(self.conf_mtx) > 0:
            for i in self.conf_mtx:
                self.conf_mtx[i].reset()

    def update_metric(self, task_pred, task_gt, task_loss):
        """
        Update batch-wise metric for each task.
            :param task_pred: [TASK_PRED1, TASK_PRED2, ...]
            :param task_gt: {'TASK_ID1': TASK_GT1, 'TASK_ID2': TASK_GT2, ...}
            :param task_loss: [TASK_LOSS1, TASK_LOSS2, ...]
        """
        curr_bs = task_pred[0].shape[0]
        r = self.data_counter / (self.data_counter + curr_bs / self.batch_size)
        e = self.epoch_counter
        self.data_counter += 1

        with torch.no_grad():
            for loss, pred, (task_id, gt) in zip(task_loss, task_pred, task_gt.items()):
                self.metric[task_id][e, 0] = r * self.metric[task_id][e, 0] + (1 - r) * loss.item()

                if task_id in ['seg', 'part_seg']:
                    # update confusion matrix (metric will be computed directly in the Confusion Matrix)
                    self.conf_mtx[task_id].update(pred.argmax(1).flatten(), gt.flatten())

                if 'class' in task_id:
                    # Accuracy for image classification tasks
                    pred_label = pred.data.max(1)[1]
                    acc = pred_label.eq(gt).sum().item() / pred_label.shape[0]
                    self.metric[task_id][e, 1] = r * self.metric[task_id][e, 1] + (1 - r) * acc

                if task_id in ['depth', 'disp', 'noise']:
                    # Abs. Err.
                    invalid_idx = -1 if task_id == 'disp' else 0
                    valid_mask = (torch.sum(gt, dim=1, keepdim=True) != invalid_idx).to(pred.device)
                    abs_err = torch.mean(torch.abs(pred - gt).masked_select(valid_mask)).item()
                    self.metric[task_id][e, 1] = r * self.metric[task_id][e, 1] + (1 - r) * abs_err

                if task_id in ['normal']:
                    # Mean Degree Err.
                    valid_mask = (torch.sum(gt, dim=1) != 0).to(pred.device)
                    degree_error = torch.acos(torch.clamp(torch.sum(pred * gt, dim=1).masked_select(valid_mask), -1, 1))
                    mean_error = torch.mean(torch.rad2deg(degree_error)).item()
                    self.metric[task_id][e, 1] = r * self.metric[task_id][e, 1] + (1 - r) * mean_error

    def compute_metric(self, only_pri=False):
        metric_str = ''
        e = self.epoch_counter
        tasks = self.pri_tasks if only_pri else self.train_tasks  # only print primary tasks performance in evaluation

        for task_id in tasks:
            if task_id in ['seg', 'part_seg']:  # mIoU for segmentation
                self.metric[task_id][e, 1] = self.conf_mtx[task_id].get_metrics()

            metric_str += ' {} {:.4f} {:.4f}'\
                .format(task_id.capitalize(), self.metric[task_id][e, 0], self.metric[task_id][e, 1])

        if self.include_mtl:
            # Pre-computed single task learning performance using trainer_dense_single.py
            if self.dataset == 'nyuv2':
                stl = {'seg': 0.4337, 'depth': 0.5224, 'normal': 22.40}
            elif self.dataset == 'cityscapes':
                stl = {'seg': 0.5620, 'part_seg': 0.5274, 'disp': 0.84}
            elif self.dataset == 'cifar100':
                stl = {'class_0': 0.6865, 'class_1': 0.8100, 'class_2': 0.8234, 'class_3': 0.8371, 'class_4': 0.8910,
                       'class_5': 0.8872, 'class_6': 0.8475, 'class_7': 0.8588, 'class_8': 0.8707, 'class_9': 0.9015,
                       'class_10': 0.8976, 'class_11': 0.8488, 'class_12': 0.9033, 'class_13': 0.8441, 'class_14': 0.5537,
                       'class_15': 0.7584, 'class_16': 0.7279, 'class_17': 0.7537, 'class_18': 0.9148, 'class_19': 0.9469}

            delta_mtl = 0
            for task_id in self.train_tasks:
                if task_id in ['seg', 'part_seg']:  # higher better
                    delta_mtl += (self.metric[task_id][e, 1] - stl[task_id]) / stl[task_id]
                elif task_id in ['depth', 'normal', 'disp']:   # lower better
                    delta_mtl -= (self.metric[task_id][e, 1] - stl[task_id]) / stl[task_id]
                elif 'class' in task_id: 
                    delta_mtl += (self.metric[task_id][e, 1]) 


            self.metric['all'][e] = delta_mtl / len(stl)
            metric_str += ' | All {:.4f}'.format(self.metric['all'][e])
        return metric_str

    def get_best_performance(self, task):
        e = self.epoch_counter
        if task in ['seg', 'part_seg'] or 'class' in task:  # higher better
            return max(self.metric[task][:e, 1])
        if task in ['depth', 'normal', 'disp']:   # lower better
            return min(self.metric[task][:e, 1])
        if task in ['all']:  # higher better
            return max(self.metric[task][:e])


"""
Define Gradient-based frameworks here. 
Based on https://github.com/Cranial-XIX/CAGrad/blob/main/cityscapes/utils.py
"""


def graddrop(grads):
    P = 0.5 * (1. + grads.sum(1) / (grads.abs().sum(1) + 1e-8))
    U = torch.rand_like(grads[:, 0])
    M = P.gt(U).view(-1, 1) * grads.gt(0) + P.lt(U).view(-1, 1) * grads.lt(0)
    g = (grads * M.float()).mean(1)
    return g


def pcgrad(grads, rng, num_tasks):
    grad_vec = grads.t()

    shuffled_task_indices = np.zeros((num_tasks, num_tasks - 1), dtype=int)
    for i in range(num_tasks):
        task_indices = np.arange(num_tasks)
        task_indices[i] = task_indices[-1]
        shuffled_task_indices[i] = task_indices[:-1]
        rng.shuffle(shuffled_task_indices[i])
    shuffled_task_indices = shuffled_task_indices.T

    normalized_grad_vec = grad_vec / (grad_vec.norm(dim=1, keepdim=True) + 1e-8)  # num_tasks x dim
    modified_grad_vec = deepcopy(grad_vec)
    for task_indices in shuffled_task_indices:
        normalized_shuffled_grad = normalized_grad_vec[task_indices]  # num_tasks x dim
        dot = (modified_grad_vec * normalized_shuffled_grad).sum(dim=1, keepdim=True)   # num_tasks x dim
        modified_grad_vec -= torch.clamp_max(dot, 0) * normalized_shuffled_grad
    g = modified_grad_vec.mean(dim=0)
    return g


def cagrad(grads, num_tasks, alpha=0.5, rescale=1):
    GG = grads.t().mm(grads).cpu()  # [num_tasks, num_tasks]
    g0_norm = (GG.mean() + 1e-8).sqrt()  # norm of the average gradient

    x_start = np.ones(num_tasks) / num_tasks
    bnds = tuple((0, 1) for x in x_start)
    cons = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})
    A = GG.numpy()
    b = x_start.copy()
    c = (alpha * g0_norm + 1e-8).item()

    def objfn(x):
        return (x.reshape(1, num_tasks).dot(A).dot(b.reshape(num_tasks, 1)) + c * np.sqrt(
            x.reshape(1, num_tasks).dot(A).dot(x.reshape(num_tasks, 1)) + 1e-8)).sum()

    res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
    w_cpu = res.x
    ww = torch.Tensor(w_cpu).to(grads.device)
    gw = (grads * ww.view(1, -1)).sum(1)
    gw_norm = gw.norm()
    lmbda = c / (gw_norm + 1e-8)
    g = grads.mean(1) + lmbda * gw
    if rescale == 0:
        return g
    elif rescale == 1:
        return g / (1 + alpha ** 2)
    else:
        return g / (1 + alpha)


def grad2vec(m, grads, grad_dims, task):
    # store the gradients
    grads[:, task].fill_(0.0)
    cnt = 0
    for mm in m.shared_modules():
        for p in mm.parameters():
            grad = p.grad
            if grad is not None:
                grad_cur = grad.data.detach().clone()
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[:cnt + 1])
                grads[beg:en, task].copy_(grad_cur.data.view(-1))
            cnt += 1


def overwrite_grad(m, newgrad, grad_dims, num_tasks):
    newgrad = newgrad * num_tasks
    cnt = 0
    for mm in m.shared_modules():
        for param in mm.parameters():
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(param.data.size())
            param.grad = this_grad.data.clone()
            cnt += 1

def project_simplex(v, z=1.0, axis=-1):
    """
    Implements the algorithm in Figure 1 of
    John Duchi, Shai Shalev-Shwartz, Yoram Singer, Tushar Chandra,
    "Efficient Projections onto the l1-Ball for Learning in High Dimensions", ICML 2008.
    https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf
    This algorithm project vectors v onto the simplex w >= 0, \sum w_i = z.
    :param v: A torch tensor, will be interpreted as a collection of vectors.
    :param z: Vectors will be projected onto the z-Simplex: \sum w_i = z.
    :param axis: Indicates the axis of v, which defines the vectors to be projected.
    :return: w: result of the projection
    """

    def _project_simplex_2d(v, z):
        """
        Helper function, assuming that all vectors are arranged in rows of v.
        :param v: NxD torch tensor; Duchi et al. algorithm is applied to each row in vecotrized form
        :param z: Vectors will be projected onto the z-Simplex: \sum w_i = z.
        :return: w: result of the projection
        """
        with torch.no_grad():
            shape = v.shape
            if shape[1] == 1:
                w = v.clone().detach()
                w[:] = z
                return w

            mu = torch.sort(v, dim=1)[0]
            mu = torch.flip(mu, dims=(1,))
            cum_sum = torch.cumsum(mu, dim=1)
            j = torch.unsqueeze(torch.arange(1, shape[1] + 1, dtype=mu.dtype, device=mu.device), 0)
            rho = torch.sum(mu * j - cum_sum + z > 0.0, dim=1, keepdim=True) - 1
            max_nn = cum_sum[torch.arange(shape[0]), rho[:, 0]]
            theta = (torch.unsqueeze(max_nn, -1) - z) / (rho.type(max_nn.dtype) + 1)
            w = torch.clamp(v - theta, min=0.0)
            return w

    with torch.no_grad():
        shape = v.shape

        if len(shape) == 1:
            return _project_simplex_2d(torch.unsqueeze(v, 0), z)[0, :]
        else:
            axis = axis % len(shape)
            t_shape = tuple(range(axis)) + tuple(range(axis + 1, len(shape))) + (axis,)
            tt_shape = tuple(range(axis)) + (len(shape) - 1,) + tuple(range(axis, len(shape) - 1))
            v_t = v.permute(t_shape)
            v_t_shape = v_t.shape
            v_t_unroll = torch.reshape(v_t, (-1, v_t_shape[-1]))

            w_t = _project_simplex_2d(v_t_unroll, z)

            w_t_reroll = torch.reshape(w_t, v_t_shape)
            return w_t_reroll.permute(tt_shape)


def hard_prune(opt, prune_ratios=0.99, model=None):
    """
    hard_pruning, or direct masking
    Args:
         model: contains weight tensors in cuda
    """

    print("Hard pruning...")

    for name, weight in model.named_parameters():
        # print(name)
        if(len(weight.size()) == 4) and 'classifier' not in name:
            # if opt.method == 'mest':
            #     _, cuda_pruned_weights = weight_pruning_mest(opt, weight, prune_ratios)  # get sparse model in cuda
            # elif opt.method == 'default': 
            _, cuda_pruned_weights = weight_pruning(opt, weight, prune_ratios)
            weight.data = cuda_pruned_weights  # replace the data field in variable


def weight_pruning(opt, weight, prune_ratio):
    """
    weight pruning [irregular,column,filter]
    Args:
         weight (pytorch tensor): weight tensor, ordered by output_channel, intput_channel, kernel width and kernel height
         prune_ratio (float between 0-1): target sparsity of weights
    Returns:
         mask for nonzero weights used for retraining
         a pytorch tensor whose elements/column/row that have lowest l2 norms(equivalent to absolute weight here) are set to zero
    """


    device = weight.device
    weight = weight.cpu().detach().numpy()  # convert cpu tensor to numpy
    percent = prune_ratio * 100

    if (opt.sparsity_type == "irregular"):

        weight_temp = np.abs(weight)  # a buffer that holds weights with absolute values
        percentile = np.percentile(weight_temp, percent)  # get a value for this percentitle
        under_threshold = weight_temp < percentile
        above_threshold = weight_temp > percentile
        above_threshold = above_threshold.astype(
            np.float32)  # has to convert bool to float32 for numpy-tensor conversion
        weight[under_threshold] = 0
        return torch.from_numpy(above_threshold).cuda().to(device), torch.from_numpy(weight).cuda().to(device)
    # if (opt.sparsity_type == "irregular"):

    #     _, prune_indices = torch.topk(torch.abs(weight.flatten()),prune_ratio, largest=False)
    #     weight.data.view(weight.data.numel())[prune_indices] = 0
    #     return weight.cuda(device)
    elif (opt.sparsity_type == "filter"):
        shape = weight.shape
        weight2d = weight.reshape(shape[0], -1)
        shape2d = weight2d.shape
        row_l2_norm = LA.norm(weight2d, 2, axis=1)
        percentile = np.percentile(row_l2_norm, percent)
        under_threshold = row_l2_norm <= percentile
        above_threshold = row_l2_norm > percentile
        weight2d[under_threshold, :] = 0
        # weight2d[weight2d < 1e-40] = 0
        above_threshold = above_threshold.astype(np.float32)
        expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
        for i in range(shape2d[0]):
            expand_above_threshold[i, :] = above_threshold[i]
        weight = weight.reshape(shape)
        expand_above_threshold = expand_above_threshold.reshape(shape)
        return torch.from_numpy(expand_above_threshold).cuda().to(device), torch.from_numpy(weight).cuda().to(device)
    

def weight_pruning_mest(opt, w, prune_ratio):
    """
    weight pruning [irregular,column,filter]
    Args:
         weight (pytorch tensor): weight tensor, ordered by output_channel, intput_channel, kernel width and kernel height
         prune_ratio (float between 0-1): target sparsity of weights

    Returns:
         mask for nonzero weights used for retraining
         a pytorch tensor whose elements/column/row that have lowest l2 norms(equivalent to absolute weight here) are set to zero

    """

    device = w.device
    weight = w.detach().clone().cpu().numpy()  # convert cpu tensor to numpy
    weight_ori = copy.copy(weight)
    percent = prune_ratio * 100

    w_grad = None
    if not w.grad is None and opt.sp_lmd:
        grad_copy = copy.copy(w.grad)
        # print(grad_copy)
        w_grad = grad_copy.detach().clone().cpu().numpy()


    if (opt.sparsity_type == "irregular"):

        weight_temp = np.abs(weight)  # a buffer that holds weights with absolute values
        # print(w_grad)
        if not w_grad is None:
            grad_temp = np.abs(w_grad)
            imp_temp = weight_temp + (opt.sp_lmd * grad_temp)
        else:
            imp_temp = weight_temp
        percentile = np.percentile(imp_temp, percent)  # get a value for this percentitle
        under_threshold = imp_temp < percentile
        above_threshold = imp_temp > percentile
        above_threshold = above_threshold.astype(
            np.float32)  # has to convert bool to float32 for numpy-tensor conversion
        # weight[under_threshold] = 0
        ww = weight * above_threshold
        return torch.from_numpy(above_threshold).cuda().to(device), torch.from_numpy(ww).cuda().to(device)


def layer_grow(model, masks, layer_grow_ratios):

    print("Layer growing using top-K method...")

    for name, weight in (model.named_parameters()):
        if(len(weight.size()) == 4) and 'classifier' not in name:
            
            # weight = weight.cpu()
            freeze_mask = masks[name]
            # #values = list(freeze_mask.values())
            # values = freeze_mask
            # vector = torch.cat([value.flatten() for value in values])
            flattened_vector = freeze_mask.flatten()
            invalid_indices = torch.nonzero(flattened_vector, as_tuple=False).view(-1)
            n_grow = math.ceil(weight.data.numel() * layer_grow_ratios)
            # print(weight.grad)
            valid_grad = torch.abs(weight.grad.clone())

            one_dim_valid_grad = valid_grad.flatten()
            one_dim_valid_grad[invalid_indices] = torch.tensor(float('-inf')).to(device='cuda')
            #valid_grad.view(valid_grad.numel())[invalid_indices] = float('-inf')

            #v1 = valid_grad.flatten()
            _, grow_indices = torch.topk(one_dim_valid_grad ,n_grow, largest=True)
            weight.data.view(weight.data.numel())[grow_indices] = 0

            masks[name].view(masks[name].numel())[grow_indices] = 1
    
    return masks