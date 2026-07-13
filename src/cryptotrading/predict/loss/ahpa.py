"""
https://github.com/Meteor-Stars/APTF/blob/master/Amortized_Hierarchical_Predictability_Aware_Loss.py
https://arxiv.org/pdf/2602.16224
"""
import torch

def divide_list_equally(input_list: torch.Tensor, bucket_num: int) -> list[torch.Tensor]:
    """
    Divide a list of indices into equal buckets.
    
    Args:
        input_list (torch.Tensor): The list of indices to divide.
        bucket_num (int): The number of buckets to divide the list into.
    
    Returns:
        list[torch.Tensor]: A list of lists of indices.

    """
    k = len(input_list)
    base_size = k // bucket_num
    remainder = k % bucket_num
    divided_lists = []
    for i in range(bucket_num-1):
        divided_lists.append(input_list[i * base_size:(i + 1) * base_size])
    divided_lists.append(input_list[(bucket_num-1) * base_size:])

    return divided_lists

def generate_weights(k: int) -> list[float]:
    """
    Generate weights for the buckets.
    
    Args:
        k (int): The number of buckets.
    
    Returns:
        list[float]: A list of weights.
    """
    if k <= 1:
        raise ValueError("k shoubl be bigger than 1")
    interval = 1 / (k - 1)
    result = [1 - i * interval for i in range(k)]
    result[-1] = result[-2]/2
    return result


def AHPLoss(outputs: torch.Tensor, batch_y: torch.Tensor, outputs_2: torch.Tensor, criterion: torch.nn.Module, epoch: int, args: object) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Amortized hierarchical predictability-aware_loss
    
    Args:
        outputs (torch.Tensor): The model predictions.
        batch_y (torch.Tensor): The true labels.
        outputs_2 (torch.Tensor): The second model predictions (for amortized loss).
        criterion (torch.nn.Module): The loss function.
        epoch (int): The current epoch.
        args (object): The arguments object containing training parameters.
    
    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing the loss for the first model and the loss for the second model.
    """

    def sub_loss_TSC(outputs: torch.Tensor, batch_y: torch.Tensor, outputs_2: torch.Tensor, criterion: torch.nn.Module, epoch: int, p_n: float, args: object) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the loss for the first model (outputs) and the second model (outputs_2) using the hierarchical bucketing strategy.
        
        Args:
            outputs (torch.Tensor): The model predictions.
            batch_y (torch.Tensor): The true labels.
            outputs_2 (torch.Tensor): The second model predictions (for amortized loss).
            criterion (torch.nn.Module): The loss function.
            epoch (int): The current epoch.
            p_n (float): The penalize rate.
            args (object): The arguments object containing training parameters.
        
        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the loss for the first model and the loss for the second model.
        """

        loss_1 = criterion(outputs, batch_y)
        ind_1_sorted = torch.argsort(loss_1)

        bs=outputs.shape[0]

        normal_samples_1=ind_1_sorted[:(bs-int(p_n*bs))]
        potenital_noise_samples_1=ind_1_sorted[-int(p_n*bs):]

        divided_lists_1 = [normal_samples_1, potenital_noise_samples_1]
        weights_all = [args.weights_sub]

        if args.amortization:
            loss_2 = criterion(outputs_2, batch_y)
            ind_2_sorted = torch.argsort(loss_2)
            normal_samples_2 = ind_2_sorted[:(bs - int(p_n * bs))]
            potenital_noise_samples_2 = ind_2_sorted[-int(p_n * bs):]
            divided_lists_2=[normal_samples_2,potenital_noise_samples_2]
        if not args.amortization:
            loss_1_updated = 0
            for i, bucket in enumerate(divided_lists_1):
                if len(bucket) == 0:
                    continue
                for weights in weights_all:
                    loss_1_updated+=criterion(outputs[bucket],batch_y[bucket]).mean()*weights[i]
                loss_1_updated /= len(weights_all)
            loss_2_updated = 0
            return loss_1_updated, loss_2_updated
        else:
            loss_1_updated=0
            for i, bucket in enumerate(divided_lists_2):
                if len(bucket)==0:
                    continue
                for weights in weights_all:
                    loss_1_updated+=criterion(outputs[bucket],batch_y[bucket]).mean()*weights[i]
                loss_1_updated/=len(weights_all)
            loss_2_updated=0
            for i, bucket in enumerate(divided_lists_1):
                if len(bucket)==0:
                    continue
                for weights in weights_all:
                    loss_2_updated+=criterion(outputs_2[bucket],batch_y[bucket]).mean()*weights[i]
                loss_2_updated/=len(weights_all)
            return loss_1_updated,loss_2_updated

    def sub_loss_TSF(outputs: torch.Tensor, batch_y: torch.Tensor, outputs_2: torch.Tensor, criterion: torch.nn.Module, epoch: int, weights: list[float], args: object) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the loss for the first model (outputs) and the second model (outputs_2) using the hierarchical bucketing strategy.
        
        Args:
            outputs (torch.Tensor): The model predictions.
            batch_y (torch.Tensor): The true labels.
            outputs_2 (torch.Tensor): The second model predictions (for amortized loss).
            criterion (torch.nn.Module): The loss function.
            epoch (int): The current epoch.
            weights (list[float]): The weights for the buckets.
            args (object): The arguments object containing training parameters.
        
        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the loss for the first model and the loss for the second model.
        """

        loss_1=criterion(outputs,batch_y).mean(-1).mean(-1)

        ind_1_sorted = torch.argsort(loss_1)
        divided_lists_1=divide_list_equally(ind_1_sorted,bucket_num=len(weights))
        if args.amortization:

            loss_2 = criterion(outputs_2,batch_y).mean(-1).mean(-1)


            ind_2_sorted = torch.argsort(loss_2)

            divided_lists_2 = divide_list_equally(ind_2_sorted,bucket_num=len(weights))

        if not args.amortization:
            loss_1_updated = 0
            for i, bucket in enumerate(divided_lists_1):
                if len(bucket) == 0:
                    continue
                loss_1_updated += criterion(outputs[bucket], batch_y[bucket]).mean() * weights[i]
            loss_2_updated = 0
            return loss_1_updated, loss_2_updated
        else:
            loss_1_updated=0
            for i, bucket in enumerate(divided_lists_2):
                if len(bucket)==0:
                    continue
                loss_1_updated+=criterion(outputs[bucket],batch_y[bucket]).mean()*weights[i]

            loss_2_updated=0
            for i, bucket in enumerate(divided_lists_1):
                if len(bucket)==0:
                    continue
                loss_2_updated+=criterion(outputs_2[bucket],batch_y[bucket]).mean()*weights[i]

            return loss_1_updated,loss_2_updated
    loss_1_updated_f=0
    loss_2_updated_f=0

    if args.task == 'TSC':
        start=args.start
        end=args.end
        buckets_num_all=list(range(10))
        penalize_rates=[0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
        penalize_rates=[0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25]
        epochs_all=[i*args.epoch_inteval for i in range(len(buckets_num_all))]

        if args.epoch in epochs_all:
            args.ep_id = epochs_all.index(args.epoch)

        if not args.hierarchical_bucketing:
            "Using fixed multiple bucket groups instead of dynamically changing as training epochs evolve"
            for i,p_n in enumerate([penalize_rates[args.ep_id]]):
                loss_1_updated, loss_2_updated = sub_loss_TSC(outputs, batch_y, outputs_2, criterion, epoch, p_n, args)
                loss_1_updated_f += loss_1_updated
                loss_2_updated_f += loss_2_updated
        else:
            for i, p_n in enumerate(penalize_rates[:args.ep_id + 1]):
                loss_1_updated, loss_2_updated = sub_loss_TSC(outputs, batch_y, outputs_2, criterion, epoch, p_n, args)
                loss_1_updated_f += loss_1_updated
                loss_2_updated_f += loss_2_updated

    elif args.task == 'TSF':
        start = 1
        end = args.bucket_num_K

        buckets_num_all = list(range(start, end))
        weight_initial = generate_weights(buckets_num_all[-1])

        if not args.hierarchical_bucketing:
            epochs_all = [i * 2 for i in range(len(buckets_num_all))]
            buckets_num_all.reverse()
            if args.epoch in epochs_all:
                args.ep_id = epochs_all.index(args.epoch)
            ##don't consider previous bucketing strategy
            for i, k in enumerate(buckets_num_all[args.ep_id + 1]):
                gen_weight_tar = weight_initial[-k:]
                loss_1_updated, loss_2_updated = sub_loss_TSF(outputs, batch_y, outputs_2, criterion, epoch,
                                                          gen_weight_tar, args)
                loss_1_updated_f += loss_1_updated
                loss_2_updated_f += loss_2_updated
        else:

            epochs_all = [i * 2 for i in range(len(buckets_num_all))]
            if args.epoch in epochs_all:
                args.ep_id = epochs_all.index(args.epoch)
            ##consider previous bucketing strategy
            for i, k in enumerate(buckets_num_all[:args.ep_id + 1]):
                gen_weight_tar = weight_initial[i:]
                loss_1_updated, loss_2_updated = sub_loss_TSF(outputs, batch_y, outputs_2, criterion, epoch,
                                                          gen_weight_tar, args)
                loss_1_updated_f += loss_1_updated
                loss_2_updated_f += loss_2_updated


    return loss_1_updated_f/(end-start),loss_2_updated_f/(end-start)




