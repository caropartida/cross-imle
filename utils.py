import random

import numpy as np
import torch
from chainer import cuda


def downsample_point_cloud(points, n_pts):
    """downsample points by random choice

    :param points: (n, 3)
    :param n_pts: int
    :return:
    """
    p_idx = random.choices(list(range(points.shape[0])), k=n_pts)
    return points[p_idx]


def l2_norm(x, y):
    """Calculate l2 norm (distance) of `x` and `y`.
    Args:
        x (numpy.ndarray or cupy): (batch_size, num_point, coord_dim)
        y (numpy.ndarray): (batch_size, num_point, coord_dim)
    Returns (numpy.ndarray): (batch_size, num_point,)
    """
    return ((x - y) ** 2).sum(axis=2)

def farthest_point_sampling(pts, k, initial_idx=None, metrics=l2_norm,
                            skip_initial=False, indices_dtype=np.int32,
                            distances_dtype=np.float32):
    """Batch operation of farthest point sampling
    Code referenced from below link by @Graipher
    https://codereview.stackexchange.com/questions/179561/farthest-point-algorithm-in-python
    Args:
        pts (numpy.ndarray or cupy.ndarray): 2-dim array (num_point, coord_dim)
            or 3-dim array (batch_size, num_point, coord_dim)
            When input is 2-dim array, it is treated as 3-dim array with
            `batch_size=1`.
        k (int): number of points to sample
        initial_idx (int): initial index to start farthest point sampling.
            `None` indicates to sample from random index,s
            in this case the returned value is not deterministic.
        metrics (callable): metrics function, indicates how to calc distance.
        skip_initial (bool): If True, initial point is skipped to store as
            farthest point. It stabilizes the function output.
        xp (numpy or cupy):
        indices_dtype (): dtype of output `indices`
        distances_dtype (): dtype of output `distances`
    Returns (tuple): `indices` and `distances`.
        indices (numpy.ndarray or cupy.ndarray): 2-dim array (batch_size, k, )
            indices of sampled farthest points.
            `pts[indices[i, j]]` represents `i-th` batch element of `j-th`
            farthest point.
        distances (numpy.ndarray or cupy.ndarray): 3-dim array
            (batch_size, k, num_point)
    """
    
    pts = pts[np.newaxis ,:, :]
    
    ndim = pts.shape[2]
    if ndim == 2:
        # insert batch_size axis
        pts = pts[None, ...]
    assert ndim == 3
    xp = cuda.get_array_module(pts)
    batch_size, num_point, coord_dim = pts.shape
    indices = xp.zeros((batch_size, k, ), dtype=indices_dtype)

    # distances[bs, i, j] is distance between i-th farthest point `pts[bs, i]`
    # and j-th input point `pts[bs, j]`.
    distances = xp.zeros((batch_size, k, num_point), dtype=distances_dtype)
    if initial_idx is None:
        indices[:, 0] = xp.random.randint(len(pts))
    else:
        indices[:, 0] = initial_idx

    batch_indices = xp.arange(batch_size)
    farthest_point = pts[batch_indices, indices[:, 0]]
    # minimum distances to the sampled farthest point
    try:
        min_distances = metrics(farthest_point[:, None, :], pts)
    except Exception as e:
        import IPython; IPython.embed()

    if skip_initial:
        # Override 0-th `indices` by the farthest point of `initial_idx`
        indices[:, 0] = xp.argmax(min_distances, axis=1)
        farthest_point = pts[batch_indices, indices[:, 0]]
        min_distances = metrics(farthest_point[:, None, :], pts)

    distances[:, 0, :] = min_distances
    for i in range(1, k):
        indices[:, i] = xp.argmax(min_distances, axis=1)
        farthest_point = pts[batch_indices, indices[:, i]]
        dist = metrics(farthest_point[:, None, :], pts)
        distances[:, i, :] = dist
        min_distances = xp.minimum(min_distances, dist)

    pts = pts[:,indices,:]
    return pts[0][0]


def positional_encoding(
    tensor, num_encoding_functions=6, include_input=True, log_sampling=True
) -> torch.Tensor:
    r"""Apply positional encoding to the input.
    Args:
        tensor (torch.Tensor): Input tensor to be positionally encoded.
        encoding_size (optional, int): Number of encoding functions used to compute
            a positional encoding (default: 6).
        include_input (optional, bool): Whether or not to include the input in the
            positional encoding (default: True).
    Returns:
    (torch.Tensor): Positional encoding of the input tensor.
    """
    # TESTED
    # Trivially, the input tensor is added to the positional encoding.
    encoding = [tensor] if include_input else []
    frequency_bands = None
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0**0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    # Special case, for no positional encoding
    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=-1)


def local_directed_hausdorff(point_cloud1:torch.Tensor, point_cloud2:torch.Tensor, reduce_mean=True):
	"""

	:param point_cloud1: (B, 3, N)
	:param point_cloud2: (B, 3, M)
	:return: directed hausdorff distance, A -> B
	"""
	n_pts1 = point_cloud1.shape[2]
	n_pts2 = point_cloud2.shape[2]
	# print(point_cloud1.max(),point_cloud2.max())
	# print(point_cloud1.min(),point_cloud2.min())

	B = point_cloud1.shape[0]
	HD = []
	for pc in range(B):
		PC1 = point_cloud1[pc]
		PC2 = point_cloud2[pc]

		hd = torch.tensor(0.).to(PC1.device)
		
		x_min = PC1[0].min()
		x_d = PC1[0].max() - PC1[0].min()

		y_min = PC1[1].min()
		y_d = PC1[1].max() - PC1[1].min()

		z_min = PC1[2].min()
		z_d = PC1[2].max() - PC1[2].min()

		
		for i in range(4):
			for j in range(4):
				for k in range(4):
					

					ll = torch.tensor([x_min+0.25*i*x_d,     y_min+0.25*(j)*y_d,   z_min+0.25*(k)*z_d]).to(PC1.device)
					ur = torch.tensor([x_min+0.25*(i+1)*x_d, y_min+0.25*(j+1)*y_d, z_min+0.25*(k+1)*z_d]).to(PC1.device) 
					# print(ll,ur)
					
					inidx = torch.logical_and(ll <= PC1.transpose(1,0), PC1.transpose(1,0) <= ur)
					inidx = torch.all(inidx,dim=1)
					
					pc1 = PC1.transpose(1,0)[inidx].transpose(1,0)

					inidx = torch.logical_and(ll <= PC2.transpose(1,0), PC2.transpose(1,0) <= ur)
					inidx = torch.all(inidx,dim=1)


					pc2 = PC2.transpose(1,0)[inidx].transpose(1,0)
					# print(pc1.shape,pc2.shape,'lllll')
					

					if pc1.shape[1]>0 and pc2.shape[1]>0:
						# print(pc1.shape,pc2.shape,'f')
						pc1 = pc1.unsqueeze(0).unsqueeze(3)
						pc1 = pc1.repeat((1, 1, 1, pc2.shape[1])) # (B, 3, N, M)
						pc2 = pc2.unsqueeze(0).unsqueeze(2)
						pc2 = pc2.repeat((1, 1, pc1.shape[2], 1)) # (B, 3, N, M)

						l2_dist = torch.sqrt(torch.sum((pc1 - pc2) ** 2, dim=1)) # (B, N, M)

						shortest_dist, _ = torch.min(l2_dist, dim=2)

						hausdorff_dist, _ = torch.max(shortest_dist, dim=1) # (B, )
						# print(hausdorff_dist)
						hd += hausdorff_dist[0]
						# print(hd,'hd')
		
		HD.append(hd)	
	# print(HD,'HD')
	if reduce_mean:
		HD = torch.mean(torch.tensor(HD))

	return HD