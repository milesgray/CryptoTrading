import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from typing import List, Tuple

def find_best_matches_full_series_batch(
    train_df: pd.DataFrame, 
    context_tensor_matrix: np.ndarray, 
    test_length: int, 
    prediction_length: int, 
    pipeline, 
    top_n: int = 1
) -> List[np.ndarray]:
    """
    Finds the top_n best matching historical segments for each input context series
    using batch processing to improve efficiency.

    Args:
        train_df (pd.DataFrame): DataFrame containing the training time series data.
        context_tensor_matrix (np.ndarray): Matrix of context embeddings to match against.
        test_length (int): The length of the context window to match.
        prediction_length (int): The length of the prediction horizon.
        pipeline: The neural network pipeline used for embedding.
        top_n (int): The number of best matches to retrieve for each context.

    Returns:
        list: List of best match segments.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prepare context embeddings
    context_batch_tensor = torch.stack(context_tensor_matrix).to(device)
    target_embeddings, _ = pipeline.embed(context_batch_tensor)
    del context_batch_tensor
    torch.cuda.empty_cache()

    target_embeddings = target_embeddings.unsqueeze(1)  

    step_size = 1
    batch_size = 10

    all_errors = []
    all_indices = []
    idx = 0

    for start_idx in tqdm(range(0, len(train_df), batch_size), total=int(len(train_df)/batch_size)):
        end_idx = min(start_idx + batch_size, len(train_df))
        current_series_batch = train_df[start_idx:end_idx].copy()

        segment_list = []
        indices_list = []

        for series in current_series_batch:
            series_values = series["target"].copy()
            series_length = len(series_values)
            for i in range(0, series_length - test_length - prediction_length + 1, step_size):
                segment_values = series_values[i:i+test_length].copy()
                segment_values = segment_values.astype(float)
                segment_tensor = torch.tensor(segment_values).to(device)

                segment_list.append(segment_tensor)
                indices_list.append((idx, i))
            idx += 1

        if not segment_list:
            continue

        batch_tensor = torch.stack(segment_list).to(device)
        other_embeddings, _ = pipeline.embed(batch_tensor)
        del segment_list, batch_tensor
        torch.cuda.empty_cache()

        other_embeddings = other_embeddings.unsqueeze(0)    
        error_matrix = torch.norm(target_embeddings - other_embeddings, dim=3, p=2)
        errors = error_matrix.sum(dim=2)

        all_errors.append(errors.cpu())  
        all_indices.extend(indices_list)

        del other_embeddings, error_matrix, errors
        torch.cuda.empty_cache()

    if not all_errors:
        return []

    all_errors = torch.cat(all_errors, dim=1)  
    best_matches = []

    for target_idx, target_errors in enumerate(all_errors):
        top_n_errors, top_n_indices = torch.topk(target_errors, top_n, largest=False)
        for i in range(top_n):
            min_error_idx = top_n_indices[i].item()
            min_error = top_n_errors[i].item()
            series_idx, index = all_indices[min_error_idx]
            best_matches.append((series_idx, min_error, index))
    del target_embeddings

    future_and_last_segment = test_length + prediction_length
    best_match_segments = []
    for series_idx, min_distance, best_match_index in best_matches:
        matching_series = train_df[series_idx]["target"]
        best_match_segment = matching_series[best_match_index:(best_match_index + future_and_last_segment)]
        best_match_segments.append(best_match_segment)

    return best_match_segments


def augment_time_series(
    train_df, 
    pipeline, 
    context_tensor_matrix, 
    prediction_length, 
    top_n
) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """
    Augments the context tensor matrix by finding the best matching historical segments
    and concatenating them to the context tensors.

    Args:
        train_df (pd.DataFrame): DataFrame containing the training time series data.
        pipeline: The neural network pipeline used for embedding.
        context_tensor_matrix (np.ndarray): Matrix of context embeddings to match against.
        prediction_length (int): The length of the prediction horizon.
        top_n (int): The number of best matches to retrieve for each context.

    Returns:
        tuple: A tuple containing the augmented matrix and the mean/std values.
    """
    test_length = len(context_tensor_matrix[0])
    best_matches = find_best_matches_full_series_batch(train_df, context_tensor_matrix, test_length, prediction_length, pipeline, top_n)

    cnt = 0
    augmented_matrix = []
    mean_std_values = []
    for context_tensor in context_tensor_matrix:
        context_tensor = torch.tensor(context_tensor, dtype=torch.float32)
        elements = best_matches[cnt:cnt+top_n]
        avg_best_segment = np.mean(elements, axis=0)
        avg_segment_tensor = torch.tensor(avg_best_segment)
        
        mask = ~torch.isnan(avg_segment_tensor)
        avg_mean = avg_segment_tensor[mask].mean()
        avg_std = torch.sqrt(((avg_segment_tensor[mask] - avg_mean) ** 2).mean()) + 1e-7
        avg_segment_tensor = normalize(avg_segment_tensor, avg_mean, avg_std)

        mask = ~torch.isnan(context_tensor)
        context_mean = context_tensor[mask].mean()
        context_std = torch.sqrt(((context_tensor[mask] - context_mean) ** 2).mean()) + 1e-7
        context_tensor = normalize(context_tensor, context_mean, context_std)        

        if np.isnan(context_tensor[0].numpy()):
            for elem in context_tensor[1:]:
                if not np.isnan(elem):
                    context_start = elem
                    break
        else:
            context_start = context_tensor[0]
        if torch.isnan(context_tensor).all():
            context_start = 0  
        best_segment_start = avg_segment_tensor[-1].numpy()
       
        difference = context_start - best_segment_start
        avg_segment_tensor += difference
        
        augmented_tensor = torch.cat((avg_segment_tensor, context_tensor))
        augmented_matrix.append(augmented_tensor)
        mean_std_values.append((context_mean, context_std))

        cnt += top_n
        
    return augmented_matrix, mean_std_values    


def min_max_scale(
    tensor: torch.Tensor,
    min_val: float,
    max_val: float
) -> torch.Tensor:
    """
    Scales the tensor to the specified min and max values.

    Args:
        tensor (torch.Tensor): The tensor to scale.
        min_val (float): The minimum value to scale to.
        max_val (float): The maximum value to scale to.

    Returns:
        torch.Tensor: The scaled tensor.
    """
    tensor_min = torch.min(tensor)
    tensor_max = torch.max(tensor)
    scaled_tensor = (tensor - tensor_min) / (tensor_max - tensor_min) * (max_val - min_val) + min_val
    return scaled_tensor

def normalize(
    tensor: torch.Tensor,
    mean: float,
    std: float
) -> torch.Tensor:
    """
    Normalizes the tensor using the mean and standard deviation.

    Args:
        tensor (torch.Tensor): The tensor to normalize.
        mean (float): The mean of the tensor.
        std (float): The standard deviation of the tensor.

    Returns:
        torch.Tensor: The normalized tensor.
    """
    return (tensor - mean) / std

def denormalize_predictions(
    predictions: List[np.ndarray],
    mean_std_values: List[tuple]
) -> List[np.ndarray]:
    """
    Denormalizes the predictions using the mean and standard deviation.

    Args:
        predictions (List[np.ndarray]): List of predictions.
        mean_std_values (List[tuple]): List of mean and standard deviation values.

    Returns:
        List[np.ndarray]: List of denormalized predictions.
    """
    denormalized_predictions = []
    for idx, prediction in enumerate(predictions):
        mean, std = mean_std_values[idx]
        prediction = prediction * std + mean
        prediction = torch.nan_to_num(prediction, nan=0.0)
        denormalized_predictions.append(prediction.cpu().numpy())

    return denormalized_predictions

def normalize_context(
    context_tensor_matrix: np.ndarray
) -> tuple:
    """
    Normalizes the context tensor matrix using the mean and standard deviation.

    Args:
        context_tensor_matrix (np.ndarray): Matrix of context embeddings.

    Returns:
        tuple: A tuple containing the normalized context matrix and the mean/std values.
    """
    mean_std_values = []
    normalized_context = []
    for idx, context_tensor in enumerate(context_tensor_matrix):
        context_tensor = torch.tensor(context_tensor, dtype=torch.float32)
      
        mask = ~torch.isnan(context_tensor)
        context_mean = context_tensor[mask].mean()
        context_std = torch.sqrt(((context_tensor[mask] - context_mean) ** 2).mean()) + 1e-7
        context_tensor = normalize(context_tensor, context_mean, context_std)    
        
        normalized_context.append(context_tensor)
        mean_std_values.append((context_mean, context_std))

    return normalized_context, mean_std_values    