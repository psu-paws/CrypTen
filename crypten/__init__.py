#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

__version__ = "0.4.0"

import builtins
import copy
import logging
import os
import warnings

import crypten.common  # noqa: F401
import crypten.communicator as comm
import crypten.config  # noqa: F401
import crypten.mpc  # noqa: F401
import crypten.nn  # noqa: F401
import torch

# other imports:
from . import debug
from .config import cfg
from .cryptensor import CrypTensor

# Setup RNG generators
generators = {
    "prev": {},
    "next": {},
    "local": {},
    "global": {},
}


def init(config_file=None, party_name=None, device=None):
    """
    Initialize CrypTen. It will initialize communicator, setup party
    name for file save / load, and setup seeds for Random Number Generatiion.
    By default the function will initialize a set of RNG generators on CPU.
    If torch.cuda.is_available() returns True, it will initialize an additional
    set of RNG generators on GPU. Users can specify the GPU device the generators are
    initialized with device.

    Args:
        party_name (str): party_name for file save and load, default is None
        device (int, str, torch.device): Specify device for RNG generators on
        GPU. Must be a GPU device.
    """
    # Load config file
    if config_file is not None:
        cfg.load_config(config_file)

    # Return and raise warning if initialized
    if comm.is_initialized():
        warnings.warn("CrypTen is already initialized.", RuntimeWarning)
        return

    # Initialize communicator
    # os.environ["GLOO_SOCKET_IFNAME"] = "en0"
    comm._init(use_threads=False, init_ttp=crypten.mpc.ttp_required())

    # Setup party name for file save / load
    if party_name is not None:
        comm.get().set_name(party_name)

    # Setup seeds for Random Number Generation
    if comm.get().get_rank() < comm.get().get_world_size():
        _setup_prng()
        if crypten.mpc.ttp_required():
            crypten.mpc.provider.ttp_provider.TTPClient._init()


def get_default_cryptensor_type():
    """Gets the default type used to create `CrypTensor`s."""
    return CrypTensor.__DEFAULT_CRYPTENSOR_TYPE__


def cryptensor(*args, cryptensor_type=None, **kwargs):
    """
    Factory function to return encrypted tensor of given `cryptensor_type`. If no
    `cryptensor_type` is specified, the default type is used.
    """

    # determine CrypTensor type to use:
    if cryptensor_type is None:
        cryptensor_type = get_default_cryptensor_type()
    if cryptensor_type not in CrypTensor.__CRYPTENSOR_TYPES__:
        raise ValueError("CrypTensor type %s does not exist." % cryptensor_type)

    # create CrypTensor:
    return CrypTensor.__CRYPTENSOR_TYPES__[cryptensor_type](*args, **kwargs)

def _setup_prng():
    """
    Generate shared random seeds to generate pseudo-random sharings of
    zero. For each device, we generator four random seeds:
        "prev"  - shared seed with the previous party
        "next"  - shared seed with the next party
        "local" - seed known only to the local party (separate from torch's default seed to prevent interference from torch.manual_seed)
        "global"- seed shared by all parties

    The "prev" and "next" random seeds are shared such that each process shares
    one seed with the previous rank process and one with the next rank.
    This allows for the generation of `n` random values, each known to
    exactly two of the `n` parties.

    For arithmetic sharing, one of these parties will add the number
    while the other subtracts it, allowing for the generation of a
    pseudo-random sharing of zero. (This can be done for binary
    sharing using bitwise-xor rather than addition / subtraction)
    """
    global generators

    # Initialize RNG Generators
    for key in generators.keys():
        generators[key][torch.device("cpu")] = torch.Generator(
            device=torch.device("cpu")
        )

    if torch.cuda.is_available():
        cuda_device_names = ["cuda"]
        for i in range(torch.cuda.device_count()):
            cuda_device_names.append(f"cuda:{i}")
        cuda_devices = [torch.device(name) for name in cuda_device_names]

        for device in cuda_devices:
            for key in generators.keys():
                generators[key][device] = torch.Generator(device=device)

    # Generate random seeds for Generators
    # NOTE: Chosen seed can be any number, but we choose as a random 64-bit
    # integer here so other parties cannot guess its value. We use os.urandom(8)
    # here to generate seeds so that forked processes do not generate the same seed.

    # Generate next / prev seeds.
    seed = int.from_bytes(os.urandom(8), "big") - 2**63
    next_seed = torch.tensor(seed)

    # Create local seed - Each party has a separate local generator
    local_seed = int.from_bytes(os.urandom(8), "big") - 2**63

    # Create global generator - All parties share one global generator for sync'd rng
    global_seed = int.from_bytes(os.urandom(8), "big") - 2**63


    global_seed = torch.tensor(global_seed)
    rank = comm.get().get_rank()
    if cfg.communicator.comm_backend == "nccl":
        global_seed = global_seed.to(f"cuda:{rank}")
        next_seed = next_seed.to(f"cuda:{rank}")

    _sync_seeds(next_seed, local_seed, global_seed)


def _sync_seeds(next_seed, local_seed, global_seed):
    """
    Sends random seed to next party, recieve seed from prev. party, and broadcast global seed

    After seeds are distributed. One seed is created for each party to coordinate seeds
    across cuda devices.
    """
    global generators

    # Populated by recieving the previous party's next_seed (irecv)
    rank = comm.get().get_rank()
    prev_seed = torch.tensor([0], dtype=torch.long)
    if cfg.communicator.comm_backend == "nccl":
        prev_seed = prev_seed.to(f"cuda:{rank}")

    # Send random seed to next party, receive random seed from prev party
    world_size = comm.get().get_world_size()
    if world_size >= 2:  # Guard against segfaults when world_size == 1.
        next_rank = (rank + 1) % world_size
        prev_rank = (next_rank - 2) % world_size

        # Kiwan: NCCL has a bug (?) where the isend and irecv order has to be
        # different for rank 0 and 1 to not hang.
        if rank == 0:
            req0 = comm.get().isend(next_seed, next_rank)
            req1 = comm.get().irecv(prev_seed, src=prev_rank)
        else:
            req1 = comm.get().irecv(prev_seed, src=prev_rank)
            req0 = comm.get().isend(next_seed, next_rank)

        req0.wait()
        req1.wait()
    else:
        prev_seed = next_seed
    torch.cuda.synchronize()

    prev_seed = prev_seed.item()
    next_seed = next_seed.item()

    # Broadcase global generator - All parties share one global generator for sync'd rng
    global_seed = comm.get().broadcast(global_seed, 0).item()

    # Create one of each seed per party
    # Note: This is configured to coordinate seeds across cuda devices
    # so that we can one party per gpu. If we want to support configurations
    # where each party runs on multiple gpu's across machines, we will
    # need to modify this.
    for device in generators["prev"].keys():
        generators["prev"][device].manual_seed(prev_seed)
        generators["next"][device].manual_seed(next_seed)
        generators["local"][device].manual_seed(local_seed)
        generators["global"][device].manual_seed(global_seed)


def manual_seed(next_seed, local_seed, global_seed):
    """
    Allow users to set their random seed for testing purposes. For each device, we set three random seeds.
    Note that prev_seed is populated using next_seed
    Args:
        next_seed  - shared seed with the next party
        local_seed - seed known only to the local party (separate from torch's default seed to prevent interference from torch.manual_seed)
        global_seed - seed shared by all parties
    """
    if cfg.debug.debug_mode:
        next_seed = torch.tensor(next_seed)
        global_seed = torch.tensor(global_seed)

        _sync_seeds(next_seed, local_seed, global_seed)
    else:
        raise ValueError("User-supplied random seeds is only allowed in debug mode")


# expose classes and functions in package:
__all__ = [
    "CrypTensor",
    "generators",
    "init",
    "mpc",
    "nn",
]
