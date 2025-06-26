import torch
import torch.nn.functional as F
import numpy as np
from numpy import linalg as LA

"""
Define task metrics, loss functions, model trainer, and apgda algorithm here.
"""


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_fit(x_pred, x_output, task_type):
    device = x_pred.device

    # binary mark to mask out undefined pixel space
    binary_mask = (torch.sum(x_output, dim=1) != 0).float().unsqueeze(1).to(device)

    if task_type == 'semantic':
        # semantic loss: depth-wise cross entropy
        loss = F.nll_loss(x_pred, x_output, ignore_index=-1)

    if task_type == 'depth':
        # depth loss: l1 norm
        loss = torch.sum(torch.abs(x_pred - x_output) * binary_mask) / torch.nonzero(binary_mask, as_tuple=False).size(0)

    if task_type == 'normal':
        # normal loss: dot product
        loss = 1 - torch.sum((x_pred * x_output) * binary_mask) / torch.nonzero(binary_mask, as_tuple=False).size(0)

    return loss


# New mIoU and Acc. formula: accumulate every pixel and average across all pixels in all images
class ConfMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
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
        acc = torch.diag(h).sum() / h.sum()
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return torch.mean(iu).item(), acc.item()


def depth_error(x_pred, x_output):
    device = x_pred.device
    binary_mask = (torch.sum(x_output, dim=1) != 0).unsqueeze(1).to(device)
    x_pred_true = x_pred.masked_select(binary_mask)
    x_output_true = x_output.masked_select(binary_mask)
    abs_err = torch.abs(x_pred_true - x_output_true)
    rel_err = torch.abs(x_pred_true - x_output_true) / x_output_true
    return (torch.sum(abs_err) / torch.nonzero(binary_mask, as_tuple=False).size(0)).item(), \
           (torch.sum(rel_err) / torch.nonzero(binary_mask, as_tuple=False).size(0)).item()


def normal_error(x_pred, x_output):
    binary_mask = (torch.sum(x_output, dim=1) != 0)
    error = torch.acos(torch.clamp(torch.sum(x_pred * x_output, 1).masked_select(binary_mask), -1, 1)).detach().cpu().numpy()
    error = np.degrees(error)
    return np.mean(error), np.median(error), np.mean(error < 11.25), np.mean(error < 22.5), np.mean(error < 30)

"""
=========== Universal Multi-task Trainer =========== 
"""

def multi_task_trainer(train_loader, test_loader, multi_task_model, device, optimizer, scheduler, opt, total_epoch):
    train_batch = len(train_loader)
    test_batch = len(test_loader)
    T = opt.temp
    avg_cost = np.zeros([total_epoch, 24], dtype=np.float32)
    lambda_weight = np.ones([3, total_epoch])

    penalty2 = 1.5
    penalty3 = 3
    K = 3
    beta = 50
    gamma = 5
    rew_penalty = 1e-7

    task_W = torch.ones(K) / K

    if opt.stage == 'retrain':
        load_dir = "./checkpoints/rew.pt"
        multi_task_model.load_state_dict(torch.load(load_dir))
        hard_prune(opt, prune_ratios=opt.prune_ratios, model = multi_task_model)

        print("masked retrain")
        masks = {}
        for name, W in (multi_task_model.named_parameters()):
            if(len(W.size()) == 4) and 'task' not in name:
                weight = W.cpu().detach().numpy()
                non_zeros = weight != 0
                non_zeros = non_zeros.astype(np.float32)
                zero_mask = torch.from_numpy(non_zeros).cuda()
                W = torch.from_numpy(weight).cuda()
                W.data = W
                masks[name] = zero_mask

    if opt.stage == 'rew':
        load_dir = "./checkpoints/temp.pt"
        multi_task_model.load_state_dict(torch.load(load_dir))

    layers = []
    for name, weight in multi_task_model.named_parameters():
        if(len(weight.size()) == 4) and 'task' not in name:
            layers.append(weight)

    eps = 1e-6

    # initialize rew_layer
    rew_layers = []
    for i in range(len(layers)):
        conv_layer = layers[i]

        rew_layers.append(1 / (conv_layer.data + eps))


    for index in range(total_epoch):
        cost = np.zeros(24, dtype=np.float32)

        # apply Dynamic Weight Average
        if opt.weight == 'dwa':
            if index == 0 or index == 1:
                lambda_weight[:, index] = 1.0
            else:
                w_1 = avg_cost[index - 1, 0] / avg_cost[index - 2, 0]
                w_2 = avg_cost[index - 1, 3] / avg_cost[index - 2, 3]
                w_3 = avg_cost[index - 1, 6] / avg_cost[index - 2, 6]
                lambda_weight[0, index] = 3 * np.exp(w_1 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T) + np.exp(w_3 / T))
                lambda_weight[1, index] = 3 * np.exp(w_2 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T) + np.exp(w_3 / T))
                lambda_weight[2, index] = 3 * np.exp(w_3 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T) + np.exp(w_3 / T))

        # iteration for all batches
        multi_task_model.train()
        train_dataset = iter(train_loader)
        conf_mat = ConfMatrix(multi_task_model.class_nb)
        for k in range(train_batch):
            train_data, train_label, train_depth, train_normal = next(train_dataset)
            train_data, train_label = train_data.to(device), train_label.long().to(device)
            train_depth, train_normal = train_depth.to(device), train_normal.to(device)

            train_pred, logsigma = multi_task_model(train_data)

            optimizer.zero_grad()

            if k == 0:
                # print("reweighted l1 training...")
                rew_milestone = [60,120]
                #adjust_rew_learning_rate2(optimizer, epoch, rew_milestone, args)

            l1_loss = 0
            # add reweighted l1 loss
            if k == 0 and index in rew_milestone:
                print("reweighted l1 update")
                for j in range(len(layers)):
                    rew_layers[j] = (1 / (layers[j].data + eps))


            for j in range(len(layers)):
                rew = rew_layers[j]
                conv_layer = layers[j]

                # rew.requires_grad_(False)
                l1_loss = l1_loss + rew_penalty * torch.sum((torch.abs(rew * conv_layer)))  

            train_loss = [model_fit(train_pred[0], train_label, 'semantic'),
                          model_fit(train_pred[1], train_depth, 'depth'),
                          model_fit(train_pred[2], train_normal, 'normal')]
            
            train_loss[1] = penalty2 * train_loss[1]
            train_loss[2] = penalty3 * train_loss[2]

            if opt.weight == 'equal' or opt.weight == 'dwa':
                loss = sum([lambda_weight[i, index] * train_loss[i] for i in range(3)])
            elif opt.weight == 'minmax':
                loss = sum([task_W[i] * train_loss[i] for i in range(3)])
            else:
                loss = sum(1 / (2 * torch.exp(logsigma[i])) * train_loss[i] + logsigma[i] / 2 for i in range(3))

            if opt.stage == 'rew':
                loss = loss + l1_loss

            if opt.weight == 'minmax':

                # W = torch.from_numpy(W)
                if k == 0:

                    with torch.no_grad():
                        F = torch.stack(train_loss)
                    task_W = task_W.to(device)
                    G = F - gamma * (task_W - 1/K)
                    task_W += 1.0 / beta * G
                    task_W = project_simplex(task_W)
                    
                task_W = task_W.cpu().numpy()

            loss.backward()

            if opt.stage == 'retrain':
                with torch.no_grad():
                    for name, W in (multi_task_model.named_parameters()):
                        if name in masks:
                            W.grad *= masks[name].to(device)

            optimizer.step()

            if opt.stage == 'retrain':
                with torch.no_grad():
                    for name, W in (multi_task_model.named_parameters()):
                        if name in masks:
                            W.data *= masks[name].to(device)

            # accumulate label prediction for every pixel in training images
            conf_mat.update(train_pred[0].argmax(1).flatten(), train_label.flatten())

            cost[0] = train_loss[0].item()
            cost[3] = train_loss[1].item()
            cost[4], cost[5] = depth_error(train_pred[1], train_depth)
            cost[6] = train_loss[2].item()
            cost[7], cost[8], cost[9], cost[10], cost[11] = normal_error(train_pred[2], train_normal)
            avg_cost[index, :12] += cost[:12] / train_batch

            if opt.weight == 'minmax':
                task_W = torch.from_numpy(task_W)

        # compute mIoU and acc
        avg_cost[index, 1:3] = np.array(conf_mat.get_metrics())
        print(task_W)


        # evaluating test data
        multi_task_model.eval()
        conf_mat = ConfMatrix(multi_task_model.class_nb)
        with torch.no_grad():
            test_dataset = iter(test_loader)
            for k in range(test_batch):
                test_data, test_label, test_depth, test_normal = next(test_dataset)
                test_data, test_label = test_data.to(device), test_label.long().to(device)
                test_depth, test_normal = test_depth.to(device), test_normal.to(device)

                test_pred, _ = multi_task_model(test_data)
                test_loss = [model_fit(test_pred[0], test_label, 'semantic'),
                             model_fit(test_pred[1], test_depth, 'depth'),
                             model_fit(test_pred[2], test_normal, 'normal')]

                conf_mat.update(test_pred[0].argmax(1).flatten(), test_label.flatten())

                cost[12] = test_loss[0].item()
                cost[15] = test_loss[1].item()
                cost[16], cost[17] = depth_error(test_pred[1], test_depth)
                cost[18] = test_loss[2].item()
                cost[19], cost[20], cost[21], cost[22], cost[23] = normal_error(test_pred[2], test_normal)
                avg_cost[index, 12:] += cost[12:] / test_batch

            # compute mIoU and acc
            avg_cost[index, 13:15] = np.array(conf_mat.get_metrics())

        scheduler.step()
        print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} ||'
            'TEST: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} '
            .format(index, avg_cost[index, 0], avg_cost[index, 1], avg_cost[index, 2], avg_cost[index, 3],
                    avg_cost[index, 4], avg_cost[index, 5], avg_cost[index, 6], avg_cost[index, 7], avg_cost[index, 8],
                    avg_cost[index, 9], avg_cost[index, 10], avg_cost[index, 11], avg_cost[index, 12], avg_cost[index, 13],
                    avg_cost[index, 14], avg_cost[index, 15], avg_cost[index, 16], avg_cost[index, 17], avg_cost[index, 18],
                    avg_cost[index, 19], avg_cost[index, 20], avg_cost[index, 21], avg_cost[index, 22], avg_cost[index, 23]))

    if opt.stage == 'pretrain':
        save_dir = "./checkpoints/temp.pt"
        torch.save(multi_task_model.state_dict(), save_dir)

    if opt.stage == 'rew':
        save_dir = "./checkpoints/rew.pt"
        torch.save(multi_task_model.state_dict(), save_dir)

    if opt.stage == 'retrain':
        save_dir = "./checkpoints2/retrain.pt"
        torch.save(multi_task_model.state_dict(), save_dir)


"""
=========== Universal Single-task Trainer =========== 
"""


def single_task_trainer(train_loader, test_loader, single_task_model, device, optimizer, scheduler, opt, total_epoch=200):
    train_batch = len(train_loader)
    test_batch = len(test_loader)
    avg_cost = np.zeros([total_epoch, 24], dtype=np.float32)
    for index in range(total_epoch):
        cost = np.zeros(24, dtype=np.float32)

        # iteration for all batches
        single_task_model.train()
        train_dataset = iter(train_loader)
        conf_mat = ConfMatrix(single_task_model.class_nb)
        for k in range(train_batch):
            train_data, train_label, train_depth, train_normal = next(train_dataset)
            train_data, train_label = train_data.to(device), train_label.long().to(device)
            train_depth, train_normal = train_depth.to(device), train_normal.to(device)

            train_pred = single_task_model(train_data)
            optimizer.zero_grad()

            if opt.task == 'semantic':
                train_loss = model_fit(train_pred, train_label, opt.task)
                train_loss.backward()
                optimizer.step()

                conf_mat.update(train_pred.argmax(1).flatten(), train_label.flatten())
                cost[0] = train_loss.item()

            if opt.task == 'depth':
                train_loss = model_fit(train_pred, train_depth, opt.task)
                train_loss.backward()
                optimizer.step()
                cost[3] = train_loss.item()
                cost[4], cost[5] = depth_error(train_pred, train_depth)

            if opt.task == 'normal':
                train_loss = model_fit(train_pred, train_normal, opt.task)
                train_loss.backward()
                optimizer.step()
                cost[6] = train_loss.item()
                cost[7], cost[8], cost[9], cost[10], cost[11] = normal_error(train_pred, train_normal)

            avg_cost[index, :12] += cost[:12] / train_batch

        if opt.task == 'semantic':
            avg_cost[index, 1:3] = np.array(conf_mat.get_metrics())

        # evaluating test data
        single_task_model.eval()
        conf_mat = ConfMatrix(single_task_model.class_nb)
        with torch.no_grad():  # operations inside don't track history
            test_dataset = iter(test_loader)
            for k in range(test_batch):
                test_data, test_label, test_depth, test_normal = next(test_dataset)
                test_data, test_label = test_data.to(device),  test_label.long().to(device)
                test_depth, test_normal = test_depth.to(device), test_normal.to(device)

                test_pred = single_task_model(test_data)

                if opt.task == 'semantic':
                    test_loss = model_fit(test_pred, test_label, opt.task)

                    conf_mat.update(test_pred.argmax(1).flatten(), test_label.flatten())
                    cost[12] = test_loss.item()

                if opt.task == 'depth':
                    test_loss = model_fit(test_pred, test_depth, opt.task)
                    cost[15] = test_loss.item()
                    cost[16], cost[17] = depth_error(test_pred, test_depth)

                if opt.task == 'normal':
                    test_loss = model_fit(test_pred, test_normal, opt.task)
                    cost[18] = test_loss.item()
                    cost[19], cost[20], cost[21], cost[22], cost[23] = normal_error(test_pred, test_normal)

                avg_cost[index, 12:] += cost[12:] / test_batch
            if opt.task == 'semantic':
                avg_cost[index, 13:15] = np.array(conf_mat.get_metrics())

        scheduler.step()
        if opt.task == 'semantic':
            print('S-Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} TEST: {:.4f} {:.4f} {:.4f}'
              .format(index, avg_cost[index, 0], avg_cost[index, 1], avg_cost[index, 2], avg_cost[index, 12], avg_cost[index, 13], avg_cost[index, 14]))
        if opt.task == 'depth':
            print('D-Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} TEST: {:.4f} {:.4f} {:.4f}'
              .format(index, avg_cost[index, 3], avg_cost[index, 4], avg_cost[index, 5], avg_cost[index, 15], avg_cost[index, 16], avg_cost[index, 17]))
        if opt.task == 'normal':
            print('N-Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} TEST: {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'
              .format(index, avg_cost[index, 6], avg_cost[index, 7], avg_cost[index, 8], avg_cost[index, 9], avg_cost[index, 10], avg_cost[index, 11],
                      avg_cost[index, 18], avg_cost[index, 19], avg_cost[index, 20], avg_cost[index, 21], avg_cost[index, 22], avg_cost[index, 23]))


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

def hard_prune(opt, prune_ratios, model, option=None):
    """
    hard_pruning, or direct masking
    Args:
         model: contains weight tensors in cuda
    """

    print("hard pruning")

    for name, weight in model.named_parameters():
        if(len(weight.size()) == 4) and 'task' not in name:
            _, cuda_pruned_weights = weight_pruning(opt, weight, prune_ratios)  # get sparse model in cuda
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
