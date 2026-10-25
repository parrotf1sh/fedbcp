# 历史日志复算表

自动生成；准确率单位 %，均为每个 CSV 实际最后一条记录。峰值不用于选模。

## FedBPC / FedAvg / FedProto

| 文件 | 轮数 | 最终 | 后20轮均值 | 后20轮标准差 |
|---|---:|---:|---:|---:|
| fedavg_run_metrics.csv | 200 | 45.19 | 49.260 | 4.454 |
| fedbpc_run_metrics-1.csv | 200 | 45.90 | 47.870 | 4.872 |
| fedbpc_run_metrics-2.csv | 200 | 45.96 | 48.121 | 4.903 |
| fedbpc_smoketest_run_metrics_1.csv | 200 | 46.52 | 48.208 | 5.253 |
| fedbpc_smoketest_run_metrics_re1.csv | 200 | 46.79 | 48.390 | 4.695 |
| fedbpc_ver2.csv | 200 | 48.12 | 50.169 | 4.789 |
| fedproto_test.csv | 200 | 17.13 | 15.748 | 1.290 |
| head_false.csv | 200 | 48.61 | 50.748 | 4.507 |
| head_false_more_client.csv | 200 | 47.34 | 50.675 | 4.607 |
| head_false_mpc2.csv | 200 | 48.53 | 49.899 | 4.417 |
| head_false_proto_false.csv | 200 | 44.82 | 48.610 | 4.634 |
| head_false_re.csv | 200 | 49.07 | 49.939 | 4.528 |

FedProto 是不同预测对象（客户端原型分类器的平均）；本表不得直接充当算法排名。FedBPC 文件名不证明配置。

## FedBTR 固定最终轮测试

| 运行名 | seed | 划分指纹前12位 | Macro | Head | Middle | Tail | Worst |
|---|---:|---|---:|---:|---:|---:|---:|
| fedbtr_cifar10lt_iid_full | 2022 | 5fb90b93ce0d | 49.01 | 61.55 | 44.67 | 36.63 | 19.80 |
| fedbtr_cifar10lt_iid_natural_ce | 2022 | 5fb90b93ce0d | 36.00 | 71.97 | 18.77 | 5.27 | 0.00 |
| fedbtr_cifar10lt_iid_balanced_only | 2022 | 5fb90b93ce0d | 48.69 | 61.38 | 44.13 | 36.33 | 17.10 |
| fedbtr_cifar10lt_iid_teacher_init_only | 2022 | 5fb90b93ce0d | 45.06 | 63.68 | 40.97 | 24.33 | 9.80 |
| fedbtr_cifar10lt_iid_plain_kl | 2022 | 5fb90b93ce0d | 49.07 | 61.22 | 44.83 | 37.10 | 19.10 |
| fedbtr_cifar10lt_iid_kl_matched_relaxation | 2022 | 5fb90b93ce0d | 48.85 | 60.77 | 45.00 | 36.80 | 19.30 |
| fedbtr_cifar10lt_iid_raw_teacher_band | 2022 | 5fb90b93ce0d | 49.11 | 62.10 | 44.53 | 36.37 | 18.40 |
| fedbtr_cifar10lt_iid_uniform_support | 2022 | 5fb90b93ce0d | 49.02 | 62.10 | 45.23 | 35.37 | 16.90 |
| fedbtr_cifar10lt_iid_no_label_relaxation | 2022 | 5fb90b93ce0d | 48.84 | 61.75 | 44.20 | 36.27 | 19.00 |
| fedbtr_cifar10lt_iid_zero_band | 2022 | 5fb90b93ce0d | 48.46 | 61.35 | 43.77 | 35.97 | 18.30 |
| fedbtr_cifar10lt_full_set2 | 2022 | 177ea16e5d0f | 41.36 | 50.12 | 39.83 | 31.20 | 1.80 |
| fedbtr_cifar10lt_full_set1_trainseed2027 | 2027 | 5fb90b93ce0d | 48.45 | 62.48 | 42.80 | 35.40 | 18.70 |
| fedbtr_cifar10lt_full_set2_trainseed2027 | 2027 | 177ea16e5d0f | 42.29 | 48.70 | 44.20 | 31.83 | 1.40 |
| fedbtr_cifar10lt_full_set2_split20270817 | 2022 | f02e3ffaeb3a | 37.38 | 62.30 | 19.83 | 21.70 | 4.40 |
| fedbtr_cifar10lt_full_set1_classorder2027 | 2022 | 8eef75259388 | 47.95 | 56.40 | 44.53 | 40.10 | 33.70 |
| fedbtr_cifar10lt_natural_ce_set2_train2022 | 2022 | 177ea16e5d0f | 27.27 | 58.92 | 7.77 | 4.57 | 0.00 |
| fedbtr_cifar10lt_balanced_only_set2_train2022 | 2022 | 177ea16e5d0f | 40.38 | 46.33 | 42.83 | 30.00 | 1.40 |
| fedbtr_cifar10lt_balanced_only_set2_train2027 | 2027 | 177ea16e5d0f | 40.71 | 46.38 | 43.10 | 30.77 | 1.30 |
| fedbtr_cifar10lt_raw_teacher_band_set2_train2022 | 2022 | 177ea16e5d0f | 41.46 | 49.18 | 42.00 | 30.63 | 1.50 |
| fedbtr_cifar10lt_raw_teacher_band_set2_train2027 | 2027 | 177ea16e5d0f | 42.29 | 49.08 | 44.30 | 31.23 | 1.20 |
