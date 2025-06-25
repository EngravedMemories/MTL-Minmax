import argparse

import torch
import torch.optim as optim
import torch.utils.data.sampler as sampler

from auto_lambda import AutoLambda
from create_network import *
from create_dataset import *
from utils import *

parser = argparse.ArgumentParser(description='Multi-task Model Optimization: Sparse Prediction Tasks for NYUv2')
parser.add_argument('--mode', default='none', type=str)
parser.add_argument('--port', default='none', type=str)
parser.add_argument('--sp-lmd', type=float, default=1.0, help="importance coefficient lambda")

parser.add_argument('--network', default='mtan', type=str, help='split, mtan')
parser.add_argument('--dataset', default='nyuv2', type=str, help='nyuv2')
parser.add_argument('--weight', default='equal', type=str, help='weighting methods: autol, dwa, uncert, equal, minmax')
parser.add_argument('--stage', default='pretrain', type=str, help='training stage: pretrain, rew, retrain')
parser.add_argument('--grad_method', default='none', type=str, help='graddrop, pcgrad, cagrad')
parser.add_argument('--with_noise', action='store_true', help='with noise prediction task')
parser.add_argument('--autol_init', default=0.1, type=float, help='initialisation for auto-lambda')
parser.add_argument('--autol_lr', default=1e-4, type=float, help='learning rate for auto-lambda: `1e-4` for NYUv2')
parser.add_argument('--task', default='all', type=str, help='primary tasks, use all for MTL setting')

parser.add_argument('--sparsity-type', default='irregular', type=str, help ="define sparsity_type: [irregular,filter]")
parser.add_argument('--seed', default=0, type=int, help='random seed ID')
parser.add_argument('--gpu', default=0, type=int, help='gpu ID')
parser.add_argument('--epoch', default=200, type=int, help='total epoch')
parser.add_argument('--prune_ratios', default=0.983, type=float, help ="define pruning ratios")
parser.add_argument('--layer_prune_ratios', default=0.0051, type=float, help ="define layer pruning ratios")
parser.add_argument('--layer_grow_ratios', default=0.0051, type=float, help ="define layer growing ratios")

opt = parser.parse_args()

torch.manual_seed(opt.seed)
np.random.seed(opt.seed)
random.seed(opt.seed)

# create logging folder to store training weights and losses
if not os.path.exists('logging'):
    os.makedirs('logging')

# define model, optimiser and scheduler
device = torch.device("cuda:{}".format(opt.gpu) if torch.cuda.is_available() else "cpu")
if opt.with_noise:
    train_tasks = create_task_flags('all', opt.dataset, with_noise=True)
else:
    train_tasks = create_task_flags('all', opt.dataset, with_noise=False)

pri_tasks = create_task_flags(opt.task, opt.dataset, with_noise=False)
total_epoch = opt.epoch

K = 3
beta = 10
gamma = 5    # hyperparameter for minmax
rew_penalty = 1e-7    # penalty for reweighted

task_W = torch.ones(K) / K

train_tasks_str = ''.join(task.title() + ' + ' for task in train_tasks.keys())[:-3]
pri_tasks_str = ''.join(task.title() + ' + ' for task in pri_tasks.keys())[:-3]
print('Dataset: {} | Training Task: {} | Primary Task: {} in Multi-task / Auxiliary Learning Mode with {}'
      .format(opt.dataset.title(), train_tasks_str, pri_tasks_str, opt.network.upper()))
print('Applying Multi-task Methods: Weighting-based: {} + Gradient-based: {}'
      .format(opt.weight.title(), opt.grad_method.upper()))

if opt.network == 'split':
    model = MTLDeepLabv3(train_tasks).to(device)
elif opt.network == 'mtan':
    model = MTANDeepLabv3(train_tasks).to(device)

# choose task weighting here
if opt.weight == 'autol':
    params = model.parameters()
    autol = AutoLambda(model, device, train_tasks, pri_tasks, opt.autol_init)
    meta_weight_ls = np.zeros([total_epoch, len(train_tasks)], dtype=np.float32)
    meta_optimizer = optim.Adam([autol.meta_weights], lr=opt.autol_lr)

elif opt.weight in ['dwa', 'equal']:
    T = 2.0  # temperature used in dwa
    lambda_weight = np.ones([total_epoch, len(train_tasks)])
    params = model.parameters()

elif opt.weight == 'uncert':
    logsigma = torch.tensor([-0.7] * len(train_tasks), requires_grad=True, device=device)
    params = list(model.parameters()) + [logsigma]
    logsigma_ls = np.zeros([total_epoch, len(train_tasks)], dtype=np.float32)

elif opt.weight == 'minmax':
    lambda_weight = np.ones([total_epoch, len(train_tasks)], dtype=np.float32)
    params = model.parameters()

if opt.stage == 'pretrain' or opt.stage == 'retrain':
    optimizer = optim.SGD(params, lr=0.1, weight_decay=5e-4, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epoch)

if opt.stage == 'rew':
    milestones= [45,90,135]
    optimizer = optim.SGD(params, lr=0.01, weight_decay=5e-4, momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.2)    

# define dataset
if opt.dataset == 'nyuv2':
    dataset_path = '../data/nyuv2'
    train_set = NYUv2(root=dataset_path, train=True, augmentation=True)
    test_set = NYUv2(root=dataset_path, train=False)
    batch_size = 4

train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4
)

# a copy of train_loader with different data order, used for Auto-Lambda meta-update
if opt.weight == 'autol':
    val_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

test_loader = torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size=batch_size,
    shuffle=False
)

# apply gradient methods
if opt.grad_method != 'none':
    rng = np.random.default_rng()
    grad_dims = []
    for mm in model.shared_modules():
        for param in mm.parameters():
            grad_dims.append(param.data.numel())
    grads = torch.Tensor(sum(grad_dims), len(train_tasks)).to(device)


# Train and evaluate multi-task network
train_batch = len(train_loader)
test_batch = len(test_loader)
train_metric = TaskMetric(train_tasks, pri_tasks, batch_size, total_epoch, opt.dataset)
test_metric = TaskMetric(train_tasks, pri_tasks, batch_size, total_epoch, opt.dataset, include_mtl=True)

if opt.stage == 'retrain':
    load_dir = "./checkpoints/rew_nyuv2.pt"
    # model.load_state_dict(torch.load(load_dir,map_location='cuda:0'))
    hard_prune(opt, prune_ratios=opt.prune_ratios, model = model)

    print("masked retrain")
    masks = {}
    for name, W in (model.named_parameters()):
        if(len(W.size()) == 4) and 'classifier' not in name:
            weight = W.cpu().detach().numpy()
            non_zeros = weight != 0
            non_zeros = non_zeros.astype(np.float32)
            zero_mask = torch.from_numpy(non_zeros).cuda()
            W = torch.from_numpy(weight).cuda()
            W.data = W
            masks[name] = zero_mask


if opt.stage == 'rew':
    load_dir = "./checkpoints/temp_nyuv2.pt"
    model.load_state_dict(torch.load(load_dir,map_location="cuda:{}".format(opt.gpu)))

layers = []
for name, weight in model.named_parameters():
    if(len(weight.size()) == 4) and 'classifier' not in name:  
        layers.append(weight)

eps = 1e-6

# initialize rew_layer
rew_layers = []
for i in range(len(layers)):
    conv_layer = layers[i]

    rew_layers.append(1 / (conv_layer.data + eps))

for index in range(total_epoch):

    if opt.weight == 'minmax':
        print(task_W)

    # apply Dynamic Weight Average
    if opt.weight == 'dwa':
        if index == 0 or index == 1:
            lambda_weight[index, :] = 1.0
        else:
            w = []
            for i, t in enumerate(train_tasks):
                w += [train_metric.metric[t][index - 1, 0] / train_metric.metric[t][index - 2, 0]]
            w = torch.softmax(torch.tensor(w) / T, dim=0)
            lambda_weight[index] = len(train_tasks) * w.numpy()

    # iteration for all batches
    model.train()
    train_dataset = iter(train_loader)
    if opt.weight == 'autol':
        val_dataset = iter(val_loader)

    for k in range(train_batch):
        train_data, train_target = next(train_dataset)
        train_data = train_data.to(device)
        train_target = {task_id: train_target[task_id].to(device) for task_id in train_tasks.keys()}

        # update meta-weights with Auto-Lambda
        if opt.weight == 'autol':
            val_data, val_target = next(val_dataset)
            val_data = val_data.to(device)
            val_target = {task_id: val_target[task_id].to(device) for task_id in train_tasks.keys()}

            meta_optimizer.zero_grad()
            autol.unrolled_backward(train_data, train_target, val_data, val_target,
                                    scheduler.get_last_lr()[0], optimizer)
            meta_optimizer.step()

        # update multi-task network parameters with task weights
        optimizer.zero_grad()

        if k == 0:
            print("reweighted l1 training...")
            rew_milestone = [45,90,135]

        loss = 0
        l1_loss = 0
        # add reweighted l1 loss

        if k == 0 and index in rew_milestone:
            print("reweighted l1 update")
            for j in range(len(layers)):
                rew_layers[j] = (1 / (layers[j].data + eps))


        for j in range(len(layers)):
            rew = rew_layers[j]
            conv_layer = layers[j]
            l1_loss = l1_loss + rew_penalty * torch.sum((torch.abs(rew * conv_layer)))

        if opt.stage == 'rew':
            loss = loss + l1_loss
        
        train_pred = model(train_data)
        train_loss = [compute_loss(train_pred[i], train_target[task_id], task_id) for i, task_id in enumerate(train_tasks)]

        train_loss_tmp = [0] * len(train_tasks)

        if opt.weight in ['equal', 'dwa']:
            train_loss_tmp = [w * train_loss[i] for i, w in enumerate(lambda_weight[index])]

        if opt.weight == 'uncert':
            train_loss_tmp = [1 / (2 * torch.exp(w)) * train_loss[i] + w / 2 for i, w in enumerate(logsigma)]

        if opt.weight == 'autol':
            train_loss_tmp = [w * train_loss[i] for i, w in enumerate(autol.meta_weights)]

        if opt.weight == 'minmax':
            # W = torch.from_numpy(W)
            loss = sum([task_W[i] * train_loss[i] for i in range(3)])

            if k == 0:
                
                with torch.no_grad():
                    F = torch.stack(train_loss)
                task_W = task_W.to(device)
                G = F - gamma * (task_W - 1/K)
                task_W += 1.0 / beta * G
                task_W = project_simplex(task_W)
                    
            task_W = task_W.cpu().numpy()
            
        else:
            loss = sum(train_loss_tmp)

        loss.backward()
        
        if opt.stage == 'retrain':
            if index <= 150 and index % 10 == 0 and index != 0 and k == 0:
                hard_prune(opt, prune_ratios=opt.prune_ratios, model = model)
                for name, W in (model.named_parameters()):
                    #print(name, W)
                    if(len(W.size()) == 4) and 'classifier' not in name:
                        weight = W.cpu().detach().numpy()
                        non_zeros = weight != 0
                        non_zeros = non_zeros.astype(np.float32)
                        zero_mask = torch.from_numpy(non_zeros).cuda()
                        W = torch.from_numpy(weight).cuda()
                        W.data = W
                        masks[name] = zero_mask
                masks = layer_grow(model=model, masks=masks, layer_grow_ratios=opt.layer_grow_ratios)  
        

        if opt.stage == 'retrain':
            with torch.no_grad():
                for name, W in (model.named_parameters()):
                    if name in masks and W.grad != None:
                        W.grad *= masks[name].to(device)
        optimizer.step()

        if opt.stage == 'retrain':
            with torch.no_grad():
                for name, W in (model.named_parameters()):
                    if name in masks and W.grad != None:
                        W.data *= masks[name].to(device)
                # for name, W in (multi_task_model.named_parameters()):
                #     if name == 'encoder_block.0.0.weight':
                #         print(W)

        train_metric.update_metric(train_pred, train_target, train_loss)

        # gradient-based methods applied here:
        if opt.grad_method == "graddrop":
            for i in range(len(train_tasks)):
                train_loss_tmp[i].backward(retain_graph=True)
                grad2vec(model, grads, grad_dims, i)
                model.zero_grad_shared_modules()
            g = graddrop(grads)
            overwrite_grad(model, g, grad_dims, len(train_tasks))
            optimizer.step()

        elif opt.grad_method == "pcgrad":
            for i in range(len(train_tasks)):
                train_loss_tmp[i].backward(retain_graph=True)
                grad2vec(model, grads, grad_dims, i)
                model.zero_grad_shared_modules()
            g = pcgrad(grads, rng, len(train_tasks))
            overwrite_grad(model, g, grad_dims, len(train_tasks))
            optimizer.step()

        elif opt.grad_method == "cagrad":
            for i in range(len(train_tasks)):
                train_loss_tmp[i].backward(retain_graph=True)
                grad2vec(model, grads, grad_dims, i)
                model.zero_grad_shared_modules()
            g = cagrad(grads, len(train_tasks), 0.4, rescale=1)
            overwrite_grad(model, g, grad_dims, len(train_tasks))
            optimizer.step()

        train_metric.update_metric(train_pred, train_target, train_loss)

        if opt.weight == 'minmax':
            task_W = torch.from_numpy(task_W)


    train_str = train_metric.compute_metric()
    train_metric.reset()

    # evaluating test data
    model.eval()
    with torch.no_grad():
        test_dataset = iter(test_loader)
        for k in range(test_batch):
            test_data, test_target = next(test_dataset)
            test_data = test_data.to(device)
            test_target = {task_id: test_target[task_id].to(device) for task_id in train_tasks.keys()}

            test_pred = model(test_data)
            test_loss = [compute_loss(test_pred[i], test_target[task_id], task_id) for i, task_id in
                         enumerate(train_tasks)]

            test_metric.update_metric(test_pred, test_target, test_loss)

    test_str = test_metric.compute_metric()
    test_metric.reset()

    scheduler.step()

    print('Epoch {:04d} | TRAIN:{} || TEST:{} | Best: {} {:.4f}'
          .format(index, train_str, test_str, opt.task.title(), test_metric.get_best_performance(opt.task)))

    if opt.weight == 'autol':
        meta_weight_ls[index] = autol.meta_weights.detach().cpu()
        dict = {'train_loss': train_metric.metric, 'test_loss': test_metric.metric,
                'weight': meta_weight_ls}

        print(get_weight_str(meta_weight_ls[index], train_tasks))

    elif opt.weight in ['dwa', 'equal', 'minmax']:
        dict = {'train_loss': train_metric.metric, 'test_loss': test_metric.metric,
                'weight': lambda_weight}

        print(get_weight_str(lambda_weight[index], train_tasks))

    elif opt.weight == 'uncert':
        logsigma_ls[index] = logsigma.detach().cpu()
        dict = {'train_loss': train_metric.metric, 'test_loss': test_metric.metric,
                'weight': logsigma_ls}

        print(get_weight_str(1 / (2 * np.exp(logsigma_ls[index])), train_tasks))

    np.save('logging/mtl_dense_{}_{}_{}_{}_{}_{}_.npy'
            .format(opt.network, opt.dataset, opt.task, opt.weight, opt.grad_method, opt.seed), dict)

if opt.stage == 'pretrain':
    save_dir = "./checkpoints/temp_nyuv2.pt"
    torch.save(model.state_dict(), save_dir)

if opt.stage == 'rew':
    save_dir = "./checkpoints/rew_nyuv2.pt"
    torch.save(model.state_dict(), save_dir)

if opt.stage == 'retrain':
    save_dir = "./checkpoints/retrain_nyuv2.pt"
    torch.save(model.state_dict(), save_dir)