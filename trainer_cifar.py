import argparse
import numpy as np
import torch.optim as optim
import torch.utils.data.sampler as sampler

from auto_lambda import AutoLambda
from create_network import *
from create_dataset import *
from utils import *

parser = argparse.ArgumentParser(description='Multi-task/Auxiliary Model Optimization: CIFAR-100')
parser.add_argument('--sp-lmd', type=float, default=1.0, help="importance coefficient lambda")

parser.add_argument('--mode', default='none', type=str)
parser.add_argument('--port', default='none', type=str)
parser.add_argument('--weight', default='minmax', type=str, help='multi-task weighting: autol, dwa, uncert, equal, minmax')
parser.add_argument('--stage', default='pretrain', type=str, help='training stage: pretrain, rew, retrain')
parser.add_argument('--autol_init', default=0.1, type=float, help='initialisation for auto-lambda')
parser.add_argument('--autol_lr', default=3e-4, type=float, help='learning rate for auto-lambda')
parser.add_argument('--rew_lr', default=1e-7, type=float, help='learning rate for reweighted')
parser.add_argument('--subset_id', default=-1, type=int, help='domain id for cifar-100, -1 for MTL mode')

parser.add_argument('--gamma', default=5, type=int, help='gamma for minmax')
parser.add_argument('--penalty', default=0, type=float, help='penalty for reweighted')
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
model = MTLVGG16(num_tasks=20).to(device)
train_tasks = {'class_{}'.format(i): 5 for i in range(20)}
pri_tasks = {'class_{}'.format(opt.subset_id): 5} if opt.subset_id >= 0 else train_tasks

total_epoch = opt.epoch
batch_size = 32
K = 20
beta = 50
gamma_reg = opt.gamma
rew_penalty = opt.penalty


if opt.weight == 'autol':
    params = model.parameters()
    autol = AutoLambda(model, device, train_tasks, pri_tasks, opt.autol_init)
    meta_weight_ls = np.zeros([total_epoch, len(train_tasks)], dtype=np.float32)
    meta_optimizer = optim.Adam([autol.meta_weights], lr=opt.autol_lr)

elif opt.weight in ['dwa', 'equal']:
    T = 2.0  # temperature used in dwa
    lambda_weight = np.ones([total_epoch, len(train_tasks)], dtype=np.float32)
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
    milestones= [1000]
    optimizer = optim.SGD(params, lr=opt.rew_lr, weight_decay=5e-4, momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.2)

# define dataset
trans_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
])

trans_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
])

train_sets = [CIFAR100MTL(root='dataset', train=True, transform=trans_train, subset_id=i) for i in range(20)]
if opt.subset_id >= 0:
    test_set = CIFAR100MTL(root='dataset', train=False, transform=trans_test, subset_id=opt.subset_id)
else:
    test_sets = [CIFAR100MTL(root='dataset', train=False, transform=trans_test, subset_id=i) for i in range(20)]

train_loaders = [torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=2)
                 for train_set in train_sets]

# a copy of train_loader with different data order, used for Auto-Lambda meta-update
if opt.weight == 'autol':
    val_loaders = [torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=2)
                   for train_set in train_sets]

if opt.subset_id >= 0:
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=2)
else:
    test_loaders = [torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True, num_workers=2)
                    for test_set in test_sets]


# Train and evaluate multi-task network
if opt.subset_id >= 0:
    print('CIFAR-100 | Training Task: All Domains | Primary Task: {} in Multi-task / Auxiliary Learning Mode with VGG-16'
          .format(test_set.subset_class.title()))
else:
    print('CIFAR-100 | Training Task: All Domains | Primary Task: All Domains in Multi-task / Auxiliary Learning Mode with VGG16')

print('Applying Multi-task Methods: Weighting-based: {}'
      .format(opt.weight.title()))

train_batch = len(train_loaders[0])
test_batch = len(test_loader) if opt.subset_id >= 0 else len(test_loaders[0])
train_metric = TaskMetric(train_tasks, pri_tasks, batch_size, total_epoch, 'cifar100')
if opt.subset_id >= 0:
    test_metric = TaskMetric(train_tasks, pri_tasks, batch_size, total_epoch, 'cifar100')
else:
    test_metric = TaskMetric(train_tasks, pri_tasks, batch_size, total_epoch, 'cifar100', include_mtl=True)

train_result = []
test_result =[]

task_W = torch.ones(K) / K

if opt.stage == 'retrain':
    load_dir = "./checkpoints/temp.pt"      # load temp.pt to retrain.
    # model.load_state_dict(torch.load(load_dir,map_location='cuda:0'))
    hard_prune(opt, prune_ratios=opt.prune_ratios, model = model)    # prune and grow 

    print("masked retrain...")
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
    load_dir = "./checkpoints/temp.pt"
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

    # evaluating train data
    model.train()
    train_datasets = [iter(train_loader) for train_loader in train_loaders]

    if opt.weight == 'autol':
        val_datasets = [iter(val_loader) for val_loader in val_loaders]
    
    for k in range(train_batch):
        train_datas = []
        train_targets = {}
        for t in range(20):
            train_data, train_target = next(train_datasets[t])
            train_datas += [train_data.to(device)]
            train_targets['class_{}'.format(t)] = train_target.to(device)

        if opt.weight == 'autol':
            val_datas = []
            val_targets = {}
            for t in range(20):
                val_data, val_target = next(val_datasets[t])
                val_datas += [val_data.to(device)]
                val_targets['class_{}'.format(t)] = val_target.to(device)

            meta_optimizer.zero_grad()
            autol.unrolled_backward(train_datas, train_targets, val_datas, val_targets,
                                  scheduler.get_last_lr()[0], optimizer)
            meta_optimizer.step()

        optimizer.zero_grad()

        if k == 0:
            # print("reweighted l1 training...")
            rew_milestone = [45,90,135]

        l1_loss = 0        # add reweighted l1 loss
        if k == 0 and index in rew_milestone:
            print("reweighted l1 update")
            for j in range(len(layers)):
                rew_layers[j] = (1 / (layers[j].data + eps))


        for j in range(len(layers)):
            rew = rew_layers[j]
            conv_layer = layers[j]
            l1_loss = l1_loss + rew_penalty * torch.sum((torch.abs(rew * conv_layer)))
        

        train_pred = [model(train_data, t) for t, train_data in enumerate(train_datas)]
        train_loss = [compute_loss(train_pred[t], train_targets[task_id], task_id) for t, task_id in enumerate(train_targets)]

        if opt.weight in ['equal', 'dwa']:
            loss = sum(w * train_loss[i] for i, w in enumerate(lambda_weight[index]))

        if opt.weight == 'autol':
            loss = sum(w * train_loss[i] for i, w in enumerate(autol.meta_weights))

        if opt.weight == 'uncert':
            loss = sum(1 / (2 * torch.exp(w)) * train_loss[i] + w / 2 for i, w in enumerate(logsigma))

        if opt.weight == 'minmax':

            # W = torch.from_numpy(W)
            loss = sum([task_W[i] * train_loss[i] for i in range(20)])

            if k == 0:    
                with torch.no_grad():
                    F = torch.stack(train_loss)
                task_W = task_W.to(device)
                G = F - gamma_reg * (task_W - 1/K)
                task_W += 1.0 / beta * G
                task_W = project_simplex(task_W)
                    
            task_W = task_W.cpu().numpy()

        if opt.stage == 'rew':
            loss = loss + l1_loss            

        loss.backward()
        
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
                    if name in masks:
                        W.grad *= masks[name].to(device)

        optimizer.step()

        if opt.stage == 'retrain':
            with torch.no_grad():
                for name, W in (model.named_parameters()):
                    if name in masks:
                        W.data *= masks[name].to(device)

        train_metric.update_metric(train_pred, train_targets, train_loss)

        if opt.weight == 'minmax':
            task_W = torch.from_numpy(task_W)  

    train_str = train_metric.compute_metric(only_pri=True)
    train_metric.reset()

    # evaluating test data
    model.eval()
    with torch.no_grad():
        if opt.subset_id >= 0:
            test_dataset = iter(test_loader)
            for k in range(test_batch):
                test_data, test_target = next(test_dataset)
                test_data = test_data.to(device)
                test_target = test_target.to(device)

                test_pred = model(test_data, opt.subset_id)
                test_loss = F.cross_entropy(test_pred, test_target)

                test_metric.update_metric([test_pred], {'class_{}'.format(opt.subset_id): test_target}, [test_loss])
        else:
            test_datasets = [iter(test_loader) for test_loader in test_loaders]
            for k in range(test_batch):
                test_datas = []
                test_targets = {}
                for t in range(20):
                    test_data, test_target = next(test_datasets[t])
                    test_datas += [test_data.to(device)]
                    test_targets['class_{}'.format(t)] = test_target.to(device)
                test_pred = [model(test_data, t) for t, test_data in enumerate(test_datas)]
                test_loss = [compute_loss(test_pred[t], test_targets[task_id], task_id) for t, task_id in enumerate(test_targets)]
                test_metric.update_metric(test_pred, test_targets, test_loss)

    test_str = test_metric.compute_metric(only_pri=True)
    test_metric.reset()

    scheduler.step()


    if opt.subset_id >= 0:
        print('Epoch {:04d} | TRAIN:{} || TEST:{} | Best: {} {:.4f}'
            .format(index, train_str, test_str, test_set.subset_class.title(),
                  test_metric.get_best_performance('class_{}'.format(opt.subset_id))))
        
    else:
        print('Epoch {:04d} | TRAIN:{} || TEST:{} | Best: All {:.4f}'
            .format(index, train_str, test_str, test_metric.get_best_performance('all')))
        train_result.append(train_str)
        test_result.append(test_str)

    if opt.weight == 'autol':
        meta_weight_ls[index] = autol.meta_weights.detach().cpu()
        dict = {'train_loss': train_metric.metric, 'test_loss': test_metric.metric,
                'weight': meta_weight_ls}

        print(get_weight_str_ranked(meta_weight_ls[index], list(train_sets[0].class_dict.keys()), 5))

    if opt.weight in ['dwa', 'equal', 'minmax']:
        dict = {'train_loss': train_metric.metric, 'test_loss': test_metric.metric,
                'weight': lambda_weight}
        print(get_weight_str_ranked(lambda_weight[index], list(train_sets[0].class_dict.keys()), 5))

    if opt.weight == 'uncert':
        logsigma_ls[index] = logsigma.detach().cpu()
        dict = {'train_loss': train_metric.metric, 'test_loss': test_metric.metric,
                'weight': logsigma_ls}
        print(get_weight_str_ranked(1 / (2 * np.exp(logsigma_ls[index])), list(train_sets[0].class_dict.keys()), 5))


    # np.save('logging/mtl_cifar_{}_{}_{}.npy'.format(opt.subset_id, opt.weight, opt.seed), dict)


if opt.stage == 'pretrain':
    save_dir = "./checkpoints/temp.pt"
    torch.save(model.state_dict(), save_dir)

if opt.stage == 'rew':
    save_dir = "./checkpoints/rew.pt"
    torch.save(model.state_dict(), save_dir)


class_accuracies = {}

for epoch_output in test_result:
    # Split the epoch output by spaces
    parts = epoch_output.split()
    
    # Extract the class ID and accuracy from the parts
    class_id = parts[0]
    accuracy = float(parts[-1])
    
    # Update the maximum accuracy for the class if it already exists, otherwise add it to the dictionary
    if class_id in class_accuracies:
        class_accuracies[class_id] = max(class_accuracies[class_id], accuracy)
    else:
        class_accuracies[class_id] = accuracy

# Print the maximum accuracy for each class
for class_id, accuracy in class_accuracies.items():
    print(f"Class {class_id}: {accuracy}")
