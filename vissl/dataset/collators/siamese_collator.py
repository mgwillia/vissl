#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch


def siamese_collator(batch):
    """
    batch: [
        {"data": [img1,], "label": [lbl1, ]},        #img1
        {"data": [img2,], "label": [lbl2, ]},        #img2
        .
        .
        {"data": [imgN,], "label": [lblN, ]},        #imgN
    ]

    img{x} is a tensor of size: num_towers x C x H x W
    lbl{x} is an integer
    """
    assert "data" in batch[0], "data not found in sample"
    assert "label" in batch[0], "label not found in sample"
    num_data_sources = len(batch[0]["data"])
    batch_size = len(batch)
    data = [x["data"] for x in batch]
    labels = [x["label"] for x in batch]

    output_data, output_label = [], []
    for idx in range(num_data_sources):
        # each image is of shape: num_towers x C x H x W
        # num_towers x C x H x W -> N x num_towers x C x H x W
        idx_data = torch.stack([data[i][idx] for i in range(batch_size)])
        idx_labels = [labels[i][idx] for i in range(batch_size)]
        batch_size, num_siamese_towers, channels, height, width = idx_data.size()
        # N x num_towers x C x H x W -> (N * num_towers) x C x H x W
        idx_data = idx_data.view(
            batch_size * num_siamese_towers, channels, height, width
        )
        output_data.append(idx_data)
        should_flatten = False
        if idx_labels[0].ndim == 1:
            should_flatten = True
        idx_labels = torch.stack(idx_labels).squeeze()
        if should_flatten:
            idx_labels = idx_labels.flatten()
        output_label.append(idx_labels)
    output_batch = {}
    output_batch["data"], output_batch["label"] = output_data, output_label
    return output_batch