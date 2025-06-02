import utils
import copy
import random
import math
import torch
import numpy as np
import torch.nn as nn
from hdbscan import HDBSCAN
from pyod.models.cof import COF
from scipy.spatial.distance import euclidean
from torch.utils.data import DataLoader, Subset
import sklearn.metrics.pairwise as smp
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
from torch.nn.init import xavier_normal_, kaiming_normal_
from sklearn.metrics.pairwise import cosine_similarity
from torch.nn.utils import vector_to_parameters, parameters_to_vector


class MyDST(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

kl_loss = lambda mu, logvar: -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
recon_loss = torch.nn.MSELoss(reduction='sum')

class VAE(nn.Module):
    def __init__(self, input_dim=784, latent_dim=32, hidden_dim=64):
        super(VAE,self).__init__()
        self.fc_e1 = nn.Linear(input_dim, hidden_dim)
        self.fc_e2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.fc_d1 = nn.Linear(latent_dim, hidden_dim)
        self.fc_d2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_d3 = nn.Linear(hidden_dim, input_dim)
        self.input_dim = input_dim

    def encoder(self, x_in):
        device = next(self.parameters()).device
        x_in = x_in.to(device)

        x = F.relu(self.fc_e1(x_in.view(-1, self.input_dim)))
        x = F.relu(self.fc_e2(x))
        mean = self.fc_mean(x)
        logvar = F.softplus(self.fc_logvar(x))
        return mean, logvar

    def decoder(self, z):
        z = F.relu(self.fc_d1(z))
        z = F.relu(self.fc_d2(z))
        x_out = torch.sigmoid(self.fc_d3(z))
        return x_out.view(-1, self.input_dim)

    def sample_normal(self, mean, logvar):
        sd = torch.exp(logvar*0.5)
        e = Variable(torch.randn_like(sd))
        z = e.mul(sd).add_(mean)
        return z

    def forward(self, x_in):
        z_mean, z_logvar = self.encoder(x_in)
        z = self.sample_normal(z_mean,z_logvar)
        x_out = self.decoder(z)
        return x_out, z_mean, z_logvar

    def recon_prob(self, x_in, L=10):
        with torch.no_grad():
            x_in = torch.unsqueeze(x_in, dim=0)
            x_in = torch.sigmoid(x_in)
            device = next(self.parameters()).device
            x_in = x_in.to(device)
            mean, log_var = self.encoder(x_in)
            samples_z = []
            for i in range(L):
                z = self.sample_normal(mean, log_var)
                samples_z.append(z)
            reconstruction_prob = 0.
            for z in samples_z:
                x_logit = self.decoder(z)
                x_logit = x_logit.to(device)
                reconstruction_prob += recon_loss(x_logit, x_in).item()
            return reconstruction_prob / L



class Aggregation():
    def __init__(self, agent_data_sizes, n_params, args, writer):
        self.agent_data_sizes = agent_data_sizes
        self.writer = writer
        self.args = args
        self.history_updates = {}
        self.server_lr = args.server_lr
        self.n_params = n_params
        self.cur_round = 0

    def relu(self, x):
        return torch.relu(torch.tensor(x))

    def server_train(self, global_model, criterion, aux_loader):
        initial_global_model_params = parameters_to_vector(global_model.parameters()).detach()
        global_model.train()

        if self.args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(global_model.parameters(), lr=0.01, momentum=0.9)
        elif self.args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(global_model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08,
                                         weight_decay=0, amsgrad=False)

        for _ in range(self.args.local_ep):
            for _, (inputs, labels) in enumerate(aux_loader):
                optimizer.zero_grad()
                inputs, labels = inputs.to(device=self.args.device, non_blocking=True), \
                    labels.to(device=self.args.device, non_blocking=True)
                outputs = global_model(inputs)
                minibatch_loss = criterion(outputs, labels)
                minibatch_loss.backward()
                optimizer.step()

        with torch.no_grad():
            update = parameters_to_vector(global_model.parameters()).double() - initial_global_model_params
            return update


    def aggregate_updates(self, global_model, agent_updates_dict, agent_cur_parameters_dict,model_updates):

        lr_vector = torch.Tensor([self.server_lr] * self.n_params).to(self.args.device)
        if self.args.defense == 'FedAvg':
            aggregated_updates = self.fedavg(agent_updates_dict)

        elif self.args.defense =='Krum':
            aggregated_updates = self.krum( agent_updates_dict,3)

        elif self.args.defense =='Median':
            aggregated_updates = self.median(agent_updates_dict)

        elif self.args.defense =='RLR':
            lr_vector = self.rlr(agent_updates_dict, rlr_threshold=5)
            aggregated_updates = self.fedavg(agent_updates_dict)

        elif self.args.defense =='FoolsGold':
            aggregated_updates = self.foolsgold(agent_updates_dict)

        elif self.args.defense =='FLTrust':
            aggregated_updates = self.fltrust(global_model, agent_updates_dict, aux_samples=10)

        elif self.args.defense =='FLAME':
            aggregated_updates = self.flame(agent_updates_dict, agent_cur_parameters_dict)

        elif self.args.defense =='MultiMetrics':
            aggregated_updates = self.multi_metrics(global_model, agent_cur_parameters_dict, agent_updates_dict, p=0.3)

        elif self.args.defense =='SnowBall':
            aggregated_updates = self.snowball(model_updates, agent_updates_dict)

        elif self.args.defense =='FedDLAD':
            aggregated_updates = self.combined_aggregation(agent_updates_dict, agent_cur_parameters_dict)

        cur_global_params = parameters_to_vector(global_model.parameters())
        new_global_params = (cur_global_params + lr_vector * aggregated_updates).float()
        vector_to_parameters(new_global_params, global_model.parameters())
        return


# ========================================================================================================================


    # 1.FedAvg
    def fedavg(self, agent_updates_dict):
        sm_updates, total_data = 0, 0
        for agent_id, update in agent_updates_dict.items():
            n_agent_data = self.agent_data_sizes[agent_id]
            sm_updates += n_agent_data * update
            total_data += n_agent_data
        return sm_updates / total_data


# ========================================================================================================================


    # 2.Krum
    def krum(self, agent_updates_dict, f):
        agent_ids = list(agent_updates_dict.keys())
        updates = np.array([agent_updates_dict[agent_id].cpu().numpy() for agent_id in agent_ids])
        n_clients = len(updates)
        scores = []
        for i in range(n_clients):
            distances = []
            for j in range(n_clients):
                if i != j:
                    dist = euclidean(updates[i], updates[j])
                    distances.append(dist)
            distances.sort()
            trimmed_distances = distances[:n_clients - f - 2]
            score = sum(trimmed_distances)
            scores.append((score, i))
        scores.sort()
        best_client_index = scores[0][1]
        best_agent_id = agent_ids[best_client_index]
        aggregated_updates = agent_updates_dict[best_agent_id]
        return aggregated_updates


# ========================================================================================================================


    # 3. Median
    def median(self, agent_updates_dict):
        agent_ids = list(agent_updates_dict.keys())
        updates = np.array([agent_updates_dict[agent_id].cpu().numpy() for agent_id in agent_ids])
        updates = updates.transpose()
        median_update = np.median(updates, axis=1)
        median_update = torch.tensor(median_update).to(self.args.device)
        return median_update


# ========================================================================================================================


    # 4.RLR
    def rlr(self, agent_updates_dict, rlr_threshold):
        agent_updates_sign = [torch.sign(update) for update in agent_updates_dict.values()]
        sm_of_signs = torch.abs(sum(agent_updates_sign))
        sm_of_signs[sm_of_signs < rlr_threshold] = -self.server_lr
        sm_of_signs[sm_of_signs >= rlr_threshold] = self.server_lr
        return sm_of_signs.to(self.args.device)


# ========================================================================================================================


    # 5.FoolsGold
    def foolsgold(self, agent_updates_dict):
        total_update = 0
        agent_ids = list(agent_updates_dict.keys())
        for agent_id in agent_ids:
            update = agent_updates_dict[agent_id].cpu().numpy()
            if agent_id not in self.history_updates:
                self.history_updates[agent_id] = update
            else:
                self.history_updates[agent_id] += update

        updates = np.array([self.history_updates[agent_id] for agent_id in agent_ids])
        n_clients = len(updates)
        cs = smp.cosine_similarity(updates) - np.eye(n_clients)
        maxcs = np.max(cs, axis=1)

        # Pardoning
        for i in range(n_clients):
            for j in range(n_clients):
                if i == j:
                    continue
                if maxcs[i] < maxcs[j]:
                    cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]

        wv = 1 - (np.max(cs, axis=1))
        wv[wv > 1] = 1
        wv[wv < 0] = 0

        wv = wv / np.max(wv)
        wv[(wv == 1)] = 0.99

        wv = (np.log(wv / (1 - wv)) + 0.5)
        wv[(np.isinf(wv) + wv > 1)] = 1
        wv[(wv < 0)] = 0

        for i in range (len(agent_ids)):
            total_update += wv[i] * agent_updates_dict[agent_ids[i]]
        return total_update / len(agent_ids)


# ========================================================================================================================


    #6. FLTrust
    def fltrust(self, global_model, agent_updates_dict, aux_samples):

        criterion = nn.CrossEntropyLoss().to(self.args.device)
        aux_idxs = []
        _, val_dataset = utils.get_datasets(self.args.data)
        for i in range(self.args.num_classes):
            idxs = (val_dataset.targets == i).nonzero().flatten().tolist()
            random_idxs = random.sample(idxs, min(len(idxs), aux_samples))
            aux_idxs.extend(random_idxs)
        aux_val_set = Subset(copy.deepcopy(val_dataset), aux_idxs)
        aux_loader = DataLoader(aux_val_set, batch_size=len(aux_val_set), shuffle=False,
                                pin_memory=False)

        reference_update = self.server_train(global_model, criterion, aux_loader)
        reference_update_norm = torch.norm(reference_update)

        total_score = 0
        total_update = 0

        for agent_id, update in agent_updates_dict.items():
            agent_update_norm = torch.norm(agent_updates_dict[agent_id])

            if agent_update_norm != 0:
                scale_factor = reference_update_norm / agent_update_norm
                update = agent_updates_dict[agent_id] * scale_factor
                sim_cosine = torch.nn.functional.cosine_similarity(reference_update.unsqueeze(0), update.unsqueeze(0))
                trust_score = self.relu(sim_cosine.detach().cpu().numpy())
                score_tensor = trust_score.clone().detach().to(self.args.device)
                total_score += score_tensor
                weighted_update = score_tensor * update
                total_update += weighted_update

        return total_update / total_score


# ========================================================================================================================


    #7. FLAME
    def flame(self, agent_updates_dict, agent_cur_parameters_dict):
        total_update = 0
        update_vectors = {agent_id: update.detach().cpu().numpy().astype(np.float64)
                           for agent_id, update in agent_updates_dict.items()}

        l2_norms = [np.linalg.norm(update, ord=2) for update in update_vectors.values()]
        median_norm = np.median(l2_norms)

        parameter_vectors = {agent_id: update.detach().cpu().numpy().astype(np.float64)
                           for agent_id, update in agent_cur_parameters_dict.items()}

        parameter_matrix = np.array(list(parameter_vectors.values()))
        cosine_matrix_distance = 1 - cosine_similarity(parameter_matrix)

        hdbscan = HDBSCAN(min_cluster_size=len(cosine_matrix_distance) // 2 + 1, min_samples=1,
                          allow_single_cluster=True, metric='precomputed')
        cluster_labels = hdbscan.fit_predict(cosine_matrix_distance)

        clustered_agents = {}
        for i, label in enumerate(cluster_labels):
            agent_id = list(parameter_vectors.keys())[i]
            if label not in clustered_agents:
                clustered_agents[label] = []
            clustered_agents[label].append(agent_id)
        cluster_0 = clustered_agents.get(0, [])

        for agent_id in cluster_0:
            update_vector = update_vectors[agent_id]
            update_norm = np.linalg.norm(update_vector, ord=2)
            if update_norm > median_norm:
                scale_factor = median_norm / update_norm
                update_vector = update_vector * scale_factor
            total_update += update_vector
        print(f'ids:{cluster_0}')
        total_update = torch.tensor(total_update, dtype=torch.float32).to(self.args.device)
        total_update /= len(cluster_0)
        noise_shape = next(iter(agent_updates_dict.values())).size()
        noise = torch.normal(mean=0, std=0.001 * median_norm, size=noise_shape).to(self.args.device)
        return total_update + noise


# ========================================================================================================================


    #8. MultiMetrics
    def multi_metrics(self, global_model, agent_cur_parameters_dict, agent_updates_dict, p):
            cos_dis = []
            manhattan_dis = []
            euclidean_dis = []

            # Get global model parameters and flatten them
            global_model_params = []
            for param in global_model.parameters():
                global_model_params.append(param.detach().cpu().numpy().flatten())
            global_model_flat = np.concatenate(global_model_params)

            # For each agent's current model parameters (assumed already flattened)
            for agent_id, agent_model in agent_cur_parameters_dict.items():
                # Since agent_model is already a Tensor, no need to call parameters()
                agent_model_flat = agent_model.detach().cpu().numpy().flatten()

                # Compute Cosine distance
                cosine_distance = float(
                    (1 - np.dot(global_model_flat, agent_model_flat) / (
                            np.linalg.norm(global_model_flat) * np.linalg.norm(agent_model_flat))) ** 2
                )

                manhattan_distance = float(np.linalg.norm(global_model_flat - agent_model_flat, ord=1))
                euclidean_distance = np.linalg.norm(global_model_flat - agent_model_flat)
                cos_dis.append(cosine_distance)
                manhattan_dis.append(manhattan_distance)
                euclidean_dis.append(euclidean_distance)

            # Now we need to compute the absolute differences for each agent
            total_diff_cos = np.zeros(len(agent_cur_parameters_dict))  # For cosine distances
            total_diff_manhattan = np.zeros(len(agent_cur_parameters_dict))  # For Manhattan distances
            total_diff_euclidean = np.zeros(len(agent_cur_parameters_dict))  # For Euclidean distances

            # Calculate the absolute differences for each agent against others for each metric separately
            for i in range(len(agent_cur_parameters_dict)):
                for j in range(len(agent_cur_parameters_dict)):
                    if i != j:
                        # Calculate the absolute difference for each metric separately
                        diff_cos = np.abs(cos_dis[i] - cos_dis[j])
                        diff_manhattan = np.abs(manhattan_dis[i] - manhattan_dis[j])
                        diff_euclidean = np.abs(euclidean_dis[i] - euclidean_dis[j])

                        # Add the absolute differences to the total for each metric
                        total_diff_cos[i] += diff_cos
                        total_diff_manhattan[i] += diff_manhattan
                        total_diff_euclidean[i] += diff_euclidean

            # Combine the differences into a tri_distance matrix
            tri_distance = np.vstack([total_diff_cos, total_diff_manhattan, total_diff_euclidean]).T

            # Calculate the covariance matrix of the tri_distance
            cov_matrix = np.cov(tri_distance.T)
            # Compute the inverse of the covariance matrix
            inv_matrix = np.linalg.inv(cov_matrix)

            # Calculate the Mahalanobis distance for each agent
            ma_distances = []
            for i in range(len(agent_cur_parameters_dict)):
                t = tri_distance[i]
                ma_dis = np.dot(np.dot(t, inv_matrix), t.T)  # Mahalanobis distance
                ma_distances.append(ma_dis)

            # Now, we can use the Mahalanobis distances (ma_distances) to rank the agents
            scores = ma_distances
            # Sort agents by their Mahalanobis distance (lower score is better)
            sorted_agent_indices = np.argsort(scores)

            # Calculate the number of agents to select based on the proportion p
            num_agents_to_select = int(len(sorted_agent_indices) * p)
            # Select the agents with the lowest Mahalanobis distances based on the proportion p
            selected_agent_ids = [list(agent_cur_parameters_dict.keys())[i] for i in
                                  sorted_agent_indices[:num_agents_to_select]]

            print(f"selected agents ids: {selected_agent_ids}")
            # Aggregating updates based on selected agents
            sm_updates, total_data = 0, 0
            for agent_id, update in agent_updates_dict.items():
                if agent_id in selected_agent_ids:
                    n_agent_data = self.agent_data_sizes[
                        agent_id]  # Assuming self.agent_data_sizes is defined somewhere
                    sm_updates += n_agent_data * update
                    total_data += n_agent_data

            # Return the aggregated model update normalized by total data size
            return sm_updates / total_data


# ========================================================================================================================


    #9. SnowBall
    def aggregate(self, selected_ids, agent_updates_dict):
        sm_updates, total_data = 0, 0
        agent_ids = []
        all_agent_ids = list(agent_updates_dict.keys())
        for idx in selected_ids:
            agent_id = all_agent_ids[idx]
            agent_ids.append(agent_id)

        for agent_id, update in agent_updates_dict.items():
            if agent_id in agent_ids:
                n_agent_data = self.agent_data_sizes[agent_id]
                sm_updates += n_agent_data * update
                total_data += n_agent_data

        print(f'ids:{agent_ids}')
        return sm_updates / total_data

    def cluster(self, init_ids, data):
            clusterer = KMeans(n_clusters=len(init_ids), init=[data[i] for i in init_ids], n_init=1)
            cluster_labels = clusterer.fit_predict(data)
            return cluster_labels

    kl_loss = lambda mu, logvar: -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    recon_loss = torch.nn.MSELoss(reduction='sum')

    def _flatten_model(self, model_update, layer_list=None, ignore=None):
        k_list = []
        for k in model_update.keys():
            if ignore is not None and ignore in k:
                continue
            for target_k in layer_list:
                if target_k in k:
                    k_list.append(k)
                    break
        return torch.concat([model_update[k].flatten() for k in k_list])

    def build_dif_set(self, data):
            dif_set = []
            for i in range(len(data)):
                for j in range(len(data)):
                    if i != j:
                        dif_set.append(data[i] - data[j])
            return dif_set

    def obtain_dif(self, base, target):
            dif_set = []
            for item in base:
                if torch.sum(item - target) != 0.0:
                    dif_set.append(item - target)
                    dif_set.append(target - item)
            return dif_set

    def _init_weights(self, model, init_type):
            if init_type not in ['none', 'xavier', 'kaiming']:
                raise ValueError('init must in "none", "xavier" or "kaiming"')

            def init_func(m):
                classname = m.__class__.__name__
                if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                    if init_type == 'xavier':
                        xavier_normal_(m.weight.data, gain=1.0)
                    elif init_type == 'kaiming':
                        kaiming_normal_(m.weight.data, nonlinearity='relu')

            if init_type != 'none':
                model.apply(init_func)

    def train_vae(self, vae, data, num_epoch, latent, hidden):

            data = torch.stack(data, dim=0)
            data = torch.sigmoid(data)
            if vae is None:
                vae = VAE(input_dim=len(data[0]), latent_dim=latent, hidden_dim=hidden).to(self.args.device)
                self._init_weights(vae, 'kaiming')
            vae = vae.to(self.args.device)
            vae.train()
            train_loader = DataLoader(MyDST(data), batch_size=8, shuffle=True)
            optimizer = torch.optim.Adam(vae.parameters())
            for epoch in range(num_epoch):
                for _, x in enumerate(train_loader):
                    optimizer.zero_grad()
                    x = x.to(self.args.device)
                    recon_x, mu, logvar = vae(x)
                    recon = recon_loss(recon_x, x)
                    kl = kl_loss(mu, logvar)
                    kl = torch.mean(kl)
                    loss = recon + kl
                    loss.backward()
                    optimizer.step()

            vae = vae.cpu()
            return vae


    def snowball(self, model_updates, agent_updates_dict):

            idx_list = list(agent_updates_dict.keys())
            self.cur_round += 1
            kernels = []
            for key in model_updates[0].keys():
                kernels.append([model_updates[idx_client][key] for idx_client in range(len(model_updates))])

            cnt = [0 for _ in range(len(model_updates))]

            for idx_layer, layer_name in enumerate(model_updates[0].keys()):

                if self.args.model == 'CNN_MNIST':
                    if 'conv1' not in layer_name and 'fc2' not in layer_name:
                        continue

                elif self.args.model == 'CNN_FMNIST':
                    if 'conv1' not in layer_name and 'fc2' not in layer_name:
                        continue

                elif self.args.model == 'VGG9':
                    if 'features.0' not in layer_name  and 'classifier.3' not in layer_name:
                        continue
                else:
                    if 'conv1' not in layer_name and 'fc' not in layer_name:
                        continue

                benign_list_cur_layer = []
                score_list_cur_layer = []
                updates_kernel = [item.flatten().cpu().numpy() for item in kernels[idx_layer]]
                for idx_client in range(len(updates_kernel)):
                    ddif = [updates_kernel[idx_client] - updates_kernel[i] for i in range(len(updates_kernel))]
                    norms = np.linalg.norm(ddif, axis=1)
                    norm_rank = np.argsort(norms)
                    suspicious_idx = norm_rank[-self.args.ct:]
                    centroid_ids = [idx_client]
                    centroid_ids.extend(suspicious_idx)
            #       print(centroid_ids)
                    cluster_result = self.cluster(centroid_ids, ddif)
            #       print(f'{set(cluster_result)}')
                    score_ = calinski_harabasz_score(ddif, cluster_result)
                    benign_ids = np.argwhere(cluster_result == cluster_result[idx_client]).flatten()

                    benign_list_cur_layer.append(benign_ids)
                    score_list_cur_layer.append(score_)

                score_list_cur_layer = np.array(score_list_cur_layer)
                std_, mean_ = np.std(score_list_cur_layer), np.mean(score_list_cur_layer)
                effective_ids = np.argwhere(score_list_cur_layer > 0).flatten()
                if len(effective_ids) < int(len(score_list_cur_layer) * 0.1):
                    effective_ids = np.argsort(-score_list_cur_layer)[:int(len(score_list_cur_layer) * 0.1)]

                score_list_cur_layer = (score_list_cur_layer - np.min(score_list_cur_layer)) / (
                            np.max(score_list_cur_layer) - np.min(score_list_cur_layer))
                print('Layer', idx_layer, ' STD: {0}, Mean: {1}, STD + Mean: {2}'.format(std_, mean_, std_ + mean_))
                for idx_client in effective_ids:
                    for idx_b in benign_list_cur_layer[idx_client]:
                        cnt[idx_b] += score_list_cur_layer[idx_client]

            cnt_rank = np.argsort(-np.array(cnt))

            selected_ids = cnt_rank[:math.ceil(len(cnt_rank) * 0.1)].tolist()

            if (self.args.data == 'mnist' and self.cur_round < 100) or (self.args.data == 'fmnist' and self.cur_round < 100) or (
                self.args.data == 'cifar10' and self.cur_round < 45) or (self.args.data == 'svhn' and self.cur_round < 150):
                aggregated_updates = self.aggregate(selected_ids, agent_updates_dict)
                return aggregated_updates

            if self.args.model == 'CNN_MNIST':
                flatten_update_list = [self._flatten_model(update, layer_list=['conv1','fc2']) for update in
                                       model_updates]

            elif self.args.model == 'CNN_FMNIST':
                flatten_update_list = [self._flatten_model(update, layer_list=['conv1','fc2']) for update in
                                       model_updates]

            elif self.args.model == 'VGG9':
                flatten_update_list = [self._flatten_model(update, layer_list=['features.0', 'classifier.3']) for update in
                                       model_updates]
            else:
                flatten_update_list = [self._flatten_model(update, layer_list=['conv1', 'fc']) for update in model_updates]

            initial_round, tuning_round = self.args.vae_initial, self.args.vae_tuning
            vae = self.train_vae(None, self.build_dif_set([flatten_update_list[i] for i in selected_ids]),
                                 initial_round, latent=self.args.vae_latent, hidden=self.args.vae_hidden)

            while len(selected_ids) < int(len(idx_list) * self.args.vt):
                vae = self.train_vae(vae, self.build_dif_set([flatten_update_list[i] for i in selected_ids]),
                                     tuning_round, latent=self.args.vae_latent, hidden=self.args.vae_hidden)
                vae.eval()
                with torch.no_grad():
                    rest_ids = [i for i in range(len(flatten_update_list)) if i not in selected_ids]
                    loss_ = []
                    for idx in rest_ids:
                        m_loss = 0.
                        loss_cnt = 0
                        for dif in self.obtain_dif([flatten_update_list[i] for i in selected_ids],
                                                   flatten_update_list[idx]):
                            m_loss += vae.recon_prob(dif)
                            loss_cnt += 1
                        m_loss /= loss_cnt
                        loss_.append(m_loss)
                rank_ = np.argsort(loss_)
                selected_ids.extend(np.array(rest_ids)[rank_[:min(math.ceil(len(idx_list) * self.args.v_step),
                                                                  int(len(idx_list) * self.args.vt) - len(
                                                                      selected_ids))]])

            aggregated_updates = self.aggregate(selected_ids, agent_updates_dict)
            return aggregated_updates


#========================================================================================================================


    #10. FedDLAD
    def COF(self, agent_cur_parameters_dict):
        parameter_vectors = {agent_id: update.detach().cpu().numpy().astype(np.float64)
                             for agent_id, update in agent_cur_parameters_dict.items()}
        parameter_matrix = np.array(list(parameter_vectors.values()))

        cosine_distance = 1 - cosine_similarity(parameter_matrix)
        cof = COF(contamination=0.1, n_neighbors=24)
        cof.fit(cosine_distance)
        connectivity_distances = cof.decision_function(cosine_distance)

        agent_anomaly_scores = {agent_id: connectivity_distances[idx] for idx, agent_id in
                                enumerate(parameter_vectors.keys())}

        k1 = self.args.bg
        sorted_id = sorted(agent_anomaly_scores.items(), key=lambda x: x[1])
        benign_id = sorted_id[:k1]
        benign_id_list = [item[0] for item in benign_id]
        return benign_id_list

    def combined_aggregation(self, agent_updates_dict, agent_cur_parameters_dict):
        norms = []
        reference_ids = self.COF(agent_cur_parameters_dict)

        for agent_id, update in agent_updates_dict.items():
            if agent_id in reference_ids:
                norm = np.linalg.norm(update.detach().cpu().numpy())
                norms.append(norm)
        median_norm = np.median(norms)

        for agent_id, update in agent_updates_dict.items():
            update_norm = np.linalg.norm(update.detach().cpu().numpy())
            if update_norm > median_norm:
                scale_factor = median_norm / update_norm
                update *= scale_factor
                agent_updates_dict[agent_id] = update

        all_updates = []
        total_outlier_count = 0
        for agent_id, update in agent_updates_dict.items():
            update_vector = update.clone()
            all_updates.append(update_vector.cpu().numpy())
        all_updates = np.array(all_updates)
        q1 = np.percentile(all_updates, 25, axis=0)
        q3 = np.percentile(all_updates, 75, axis=0)
        iqr = q3 - q1
        lower_bound = q1 - self.args.iqr_scale * iqr  # 0.6
        upper_bound = q3 + self.args.iqr_scale * iqr

        dimension_outlier_count = 0
        for agent_id, update in agent_updates_dict.items():
            update_vector = update.clone().cpu().numpy()
            mask = (update_vector < lower_bound) | (update_vector > upper_bound)
            total_outlier_count += np.sum(mask)
            update_vector[mask] = -update_vector[mask]
            dimension_outlier_count += mask
            agent_updates_dict[agent_id] = torch.from_numpy(update_vector).float().to(self.args.device)

        noise_mask = (dimension_outlier_count > 0).astype(int)
        count = noise_mask.sum().item()

        reference_update, reference_data = 0, 0
        for agent_id, update in agent_updates_dict.items():
            if agent_id in reference_ids:
                reference_data += self.agent_data_sizes[agent_id]
                reference_update += self.agent_data_sizes[agent_id] * agent_updates_dict[agent_id]
        reference_update /= reference_data

        pardoned_ids, score_dict = self.secondary_filtering(reference_ids, reference_update, agent_updates_dict,
                                                            k2=self.args.pg)
        pardoned_update, score = 0, 0

        if pardoned_ids:
            for agent_id in pardoned_ids:
                pardoned_update += score_dict[agent_id] * agent_updates_dict[agent_id]
                score += score_dict[agent_id]
            pardoned_update /= score
            total_update = len(reference_ids) / len(reference_ids + pardoned_ids) * reference_update + len(
                pardoned_ids) / len(reference_ids + pardoned_ids) * pardoned_update
        else:
            total_update = reference_update
        self.print(reference_ids, pardoned_ids)
        return total_update

    def secondary_filtering(self, ref_ids, reference_update, agent_updates_dict, k2):
        score_dict = {}
        for agent_id, update in agent_updates_dict.items():
            if agent_id not in ref_ids:
                sim_cosine = \
                cosine_similarity(reference_update.cpu().numpy().reshape(1, -1), update.cpu().numpy().reshape(1, -1))[
                    0][0]
                score = self.relu(sim_cosine)
                score_dict[agent_id] = score

        filtered_score_dict = {agent_id: score for agent_id, score in score_dict.items() if score != 0}
        sorted_ids = sorted(filtered_score_dict, key=filtered_score_dict.get, reverse=True)[:k2]
        return sorted_ids, score_dict

    def print(self, reference_ids, pardoned_ids):
        print(f'PardonedID---{pardoned_ids}')
        print(f'TrustedID---{reference_ids}')
        print(f'IDs---{reference_ids + pardoned_ids}')