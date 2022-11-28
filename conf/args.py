#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id',
                        default=0,
                        type=int,
                        help='the client id')
    parser.add_argument('--rounds',
                        default=50,
                        type=int,
                        help='total communication rounds')
    parser.add_argument('--epoch',
                        default=5,
                        type=int,
                        help='number of local epochs')
    parser.add_argument('--batch_size',
                        default=256,
                        type=int,
                        help='local batch size')
    parser.add_argument('--lr',
                        default=0.01,
                        type=float,
                        help='learning rate')
    parser.add_argument('--trainer_address',
                        default='127.0.0.1:60000',
                        type=str,
                        help='init method')
    parser.add_argument('--server_address',
                        default='127.0.0.1:50000',
                        type=str,
                        help='init method')
    parser.add_argument('--sample_num',
                        default=3,
                        type=int,
                        help='local_count/sum_count')
    parser.add_argument('--input_size',
                        default=784,
                        type=int,
                        help='the number of features')
    parser.add_argument('--num_class',
                        default=10,
                        type=int,
                        help='the number of class')
    parser.add_argument('--ctx_file',
                        default='../h_e/ts_ckks.config',
                        type=str,
                        help='the ctx file location')
    args = parser.parse_args()
    return args
