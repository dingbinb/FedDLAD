import torch
import utils
import models
import copy
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from agent import Agent
from time import ctime
from agent_single import Agent_single
from options import args_parser
from aggregation import Aggregation
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import parameters_to_vector, vector_to_parameters

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    args = args_parser()
    utils.print_exp_details(args)

    file_name = f"""time:{ctime()}""" \
                + f"""s_lr:{args.server_lr}-num_cor:{args.backdoor_frac}""" \
                + f"""-backdoor_frac:{args.backdoor_frac}-pttrn:{args.trigger}"""

    writer = SummaryWriter('logs/' + file_name)
    cum_poison_acc_mean = 0

    train_dataset, val_dataset = utils.get_datasets(args.data)
    val_loader = DataLoader(copy.deepcopy(val_dataset), batch_size=args.bs, shuffle=False, pin_memory=False)

    global_model = models.get_model(args.data).to(args.device)
    criterion = nn.CrossEntropyLoss().to(args.device)

    if args.data_distribution == 'iid':
        user_groups = utils.iid_distribute_data(train_dataset, args)
    elif args.data_distribution == 'non_iid':
        user_groups = utils.non_iid_distribute_data(train_dataset, args)

    idxs = (val_dataset.targets == args.base_class).nonzero().flatten().tolist()
    poisoned_val_set = utils.DatasetSplit(val_dataset, idxs)
    utils.poison_dataset(poisoned_val_set.dataset, args, idxs, poison_all=True, record_poisoned=False)
    poisoned_val_loader = DataLoader(poisoned_val_set, batch_size=args.bs, shuffle=False, pin_memory=False)

    # ------------------------------------------------------------------------------------------------------------
    attack_pool = list(range(int(args.num_agents * args.backdoor_frac)))
    agents, agent_data_sizes = [], {}
    n_model_params = len(parameters_to_vector(global_model.parameters()))
    aggregator = Aggregation(agent_data_sizes, n_model_params, args, writer)

    for agent_id in range(0, args.num_agents):
        if args.attack == 'continuous':
            agent = Agent(agent_id, args, train_dataset, user_groups[agent_id])
        elif args.attack == 'single-shot':
            agent = Agent_single(agent_id, args, train_dataset, user_groups[agent_id])

        agent_data_sizes[agent_id] = agent.n_data
        agents.append(agent)

    for rnd in tqdm(range(1, args.train_rounds + 1)):
        attack_num = 0
        rnd_global_params = parameters_to_vector(global_model.parameters()).detach()
        agent_updates_dict = {}
        agent_cur_parameters_dict = {}
        model_updates = []

        per_round_selected_agents = int(args.num_agents * args.agent_frac)
        per_round_backdoor_agents = int(args.num_agents * args.backdoor_frac * args.agent_frac)
        total_backdoor_agents = int(args.num_agents * args.backdoor_frac)
        benign_agents = per_round_selected_agents - per_round_backdoor_agents

        total_agents = args.num_agents
        backdoor_range = range(0, total_backdoor_agents)
        benign_range = [i for i in range(total_agents) if i not in backdoor_range]

        backdoor_id = sorted(np.random.choice(backdoor_range, per_round_backdoor_agents, replace=False))
        benign_id = sorted(np.random.choice(benign_range, benign_agents, replace=False))
        combined_ids = sorted(np.concatenate((backdoor_id, benign_id)))
        total_data = 0
        for agent_id in combined_ids:
            total_data += agent_data_sizes[agent_id]

        for agent_id in combined_ids:
            if args.attack == 'single-shot':
                if agent_id in attack_pool and attack_num < 1 and rnd >= args.attack_round and (rnd % args.attack_interval == 0):
                    ds = agent_data_sizes[agent_id]
                    update, cur_parameters, layer_updates = agents[agent_id].local_train(global_model, criterion, attack=True,
                                                                          total_data=total_data, ds=ds, delta=5)
                    attack_pool.remove(agent_id)
                    attack_num += 1
                    print(f'attack id:{agent_id}')
                else:
                    update, cur_parameters, layer_updates = agents[agent_id].local_train(global_model, criterion, attack=False,
                                                                          total_data=None, ds=None, delta=None)

            elif args.attack == 'continuous':
                update, cur_parameters, layer_updates = agents[agent_id].local_train(global_model, criterion)
            agent_updates_dict[agent_id] = update
            agent_cur_parameters_dict[agent_id] = cur_parameters
            model_updates.append(layer_updates)

            vector_to_parameters(copy.deepcopy(rnd_global_params), global_model.parameters())
        aggregator.aggregate_updates(global_model, agent_updates_dict, agent_cur_parameters_dict, model_updates)

        if rnd % args.snap == 0:
            with torch.no_grad():
                val_loss, (val_acc, val_per_class_acc) = utils.get_loss_n_accuracy(global_model, criterion, val_loader,
                                                                                   args)
                writer.add_scalar('Validation/Loss', val_loss, rnd)
                writer.add_scalar('Validation/Accuracy', val_acc, rnd)

                print(f'| Val_Loss/Val_Acc:  {val_loss:.3f} / {val_acc:.3f} |')
                print(f'| Val_Per_Class_Acc: {val_per_class_acc} ')

                poison_loss, (poison_acc, _) = utils.get_loss_n_accuracy(global_model, criterion, poisoned_val_loader,
                                                                         args)
                cum_poison_acc_mean += poison_acc
                writer.add_scalar('Poison/Base_Class_Accuracy', val_per_class_acc[args.base_class], rnd)
                writer.add_scalar('Poison/Poison_Accuracy', poison_acc, rnd)
                writer.add_scalar('Poison/Poison_Loss', poison_loss, rnd)
                writer.add_scalar('Poison/Cumulative_Poison_Accuracy_Mean', cum_poison_acc_mean / rnd, rnd)
                print(f'| Poison Loss/Poison Acc: {poison_loss:.3f} / {poison_acc:.3f} |')

    print('Training has finished!')






