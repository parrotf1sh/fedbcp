"""Read-only log audit and analytical counterexamples. Standard library only.

Run from anywhere: python3 -B /absolute/path/to/reproduce_audit.py
Writes derived JSON/Markdown beside this script. Never trains or changes inputs.
"""
import csv
import hashlib
import itertools
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def values(data, key):
    return [float(r[key]) for r in data if r.get(key) not in (None, "")]


def summary(data, key):
    v = values(data, key)
    if not v:
        return None
    tail = values(data[-20:], key)
    return dict(n=len(v), first=v[0], last=v[-1], mean=statistics.mean(v),
                minimum=min(v), maximum=max(v),
                last20_mean=statistics.mean(tail) if tail else None,
                last20_sd=statistics.pstdev(tail) if tail else None)


def pearson(pairs):
    if len(pairs) < 3:
        return None
    x, y = zip(*pairs)
    mx, my = statistics.mean(x), statistics.mean(y)
    den = math.sqrt(sum((a-mx)**2 for a in x)*sum((b-my)**2 for b in y))
    return sum((a-mx)*(b-my) for a,b in pairs)/den if den else None


def softmax(z):
    e = [math.exp(v-max(z)) for v in z]
    return [v/sum(e) for v in e]


def reliability(z, y=0):
    prob = softmax(z)
    correct = max(range(len(z)), key=lambda c:z[c]) == y
    return prob[y] * (1.0 if correct else 0.5) * math.exp(-0.05*math.sqrt(sum(v*v for v in z)))


def synthetic_diagnostics():
    # Actual formula in FedBPC; this is an analytical reproduction, not torch execution.
    z = [2.0, 0.0, -1.0]
    shifted = [v+100 for v in z]
    gauge = dict(logits=z, shifted_logits=shifted, softmax=softmax(z),
                 shifted_softmax=softmax(shifted), confidence=reliability(z),
                 shifted_confidence=reliability(shifted))
    assert max(abs(a-b) for a,b in zip(softmax(z),softmax(shifted))) < 1e-12
    assert gauge['confidence'] > 0.05 > gauge['shifted_confidence']
    # For two classes and iid with-replacement minibatches, a present rare class gets 1/2
    # except an all-rare batch, where it gets 1. Exact for this stated sampling model.
    balanced = []
    for batch in [1, 10, 50, 100, 500]:
        p = 0.01
        coefficient = (1-(1-p)**batch+p**batch)/2
        balanced.append(dict(batch_size=batch, rare_prior=p,
                             expected_rare_coefficient=coefficient, target=0.5))
    # A finite population toy shows HT unbiasedness and the benefit of predictive memory.
    gradients = [0.0, 0.0, 0.0, 4.0]
    memory = [0.0, 0.0, 0.0, 3.0]
    estimates, corrected = [], []
    for subset in itertools.combinations(range(4),2):
        estimates.append(sum(gradients[k] for k in subset)/2)
        corrected.append(sum(memory)/4+sum(gradients[k]-memory[k] for k in subset)/2)
    assert abs(statistics.mean(estimates)-1) < 1e-12
    assert abs(statistics.mean(corrected)-1) < 1e-12
    varp = dict(true_mean=1.0, sample_mean=statistics.mean(estimates),
                corrected_mean=statistics.mean(corrected), sample_variance=statistics.pvariance(estimates),
                corrected_variance=statistics.pvariance(corrected),
                caveat="Predictive toy memory; not an empirical performance claim or a new variance-reduction theorem.")
    bad_memory = [0.0, 0.0, 0.0, -3.0]
    bad_corrected = [sum(bad_memory)/4 + sum(gradients[k]-bad_memory[k] for k in subset)/2
                     for subset in itertools.combinations(range(4), 2)]
    varp['anti_predictive_memory_variance'] = statistics.pvariance(bad_corrected)
    assert abs(statistics.mean(bad_corrected)-1) < 1e-12
    assert varp['anti_predictive_memory_variance'] > varp['sample_variance']
    finite_balanced=[]
    for population, rare, batch in [(100,1,50),(10000,100,50)]:
        denominator=math.comb(population,batch)
        absent=math.comb(population-rare,batch)/denominator
        all_rare=math.comb(rare,batch)/denominator if rare>=batch else 0
        finite_balanced.append(dict(population=population,rare=rare,batch_size=batch,
            expected_rare_coefficient=(1-absent+all_rare)/2,target=0.5))
    # Independent Bernoulli sampling, expected two clients. Full enumeration
    # verifies HT means and class-gradient variances; no nonlinear training claim.
    sampling_design=[]
    for probabilities in [[0.5]*4,[0.4,0.4,0.4,0.8]]:
        class_contributions=[[1/3,0],[1/3,0],[1/3,0],[0,1]]
        outcomes=[]
        for indicators in itertools.product([0,1],repeat=4):
            mass=math.prod(q if keep else 1-q for q,keep in zip(probabilities,indicators))
            estimate=[sum(keep*class_contributions[k][c]/probabilities[k]
                          for k,keep in enumerate(indicators)) for c in range(2)]
            outcomes.append((mass,estimate))
        means=[sum(mass*estimate[c] for mass,estimate in outcomes) for c in range(2)]
        variances=[sum(mass*(estimate[c]-means[c])**2 for mass,estimate in outcomes) for c in range(2)]
        assert all(abs(mean-1)<1e-12 for mean in means)
        sampling_design.append(dict(inclusion_probabilities=probabilities,
            expected_clients=sum(probabilities),means=means,variances=variances,
            worst_class_variance=max(variances)))
    missing = []
    for support in [1,2,5,10,20,48,100]:
        no = math.comb(100-support,10)/math.comb(100,10) if 100-support >= 10 else 0
        one = support*math.comb(100-support,9)/math.comb(100,10) if 100-support >= 9 else 0
        missing.append(dict(support_clients=support, no_contributor_probability=no,
                            exactly_one_probability=one, no_contributor_six_rounds=no**6))
    # Missing-class negative gradients are informative even in globally balanced data.
    # Both clients observe the identical x=1 but different labels. At p=0.5 true CE
    # gradients cancel; dropping absent-class contributions yields [-0.25,-0.25].
    negative = dict(class0_client_gradient=[-0.5,0.5], class1_client_gradient=[0.5,-0.5],
                    complete_gradient=[0.0,0.0], absent_row_masked_gradient=[-0.25,-0.25],
                    caveat="Here the masked direction is a common-logit shift; the head risk gradient is nonetheless biased. General feature vectors need not yield a common shift.")
    # Stronger distinct-feature example: x_A=1, x_B=2, both labels equally likely.
    negative['different_feature_complete_gradient']=[0.25,-0.25]
    negative['different_feature_masked_gradient']=[-0.25,-0.5]
    # Execute the repository's actual dependency-free splitter on synthetic labels.
    # Histogram/allocation evidence only: the image order and SHA are NOT actual CIFAR.
    spec = importlib.util.spec_from_file_location('audit_longtail_split',
        ROOT/'src/train_tools/preprocessing/longtail_split.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    split_checks=[]
    for partition in [dict(method='iid'),dict(method='lda',alpha=0.1)]:
        manifest=module.build_manifest([c for c in range(10) for _ in range(5000)],
            n_clients=100,partition=partition,imbalance_factor=100,
            validation_fraction=0.04,split_seed=19940817,class_order_seed=2022)
        counts=manifest['global_counts']; matrix=manifest['data_map']; total=sum(counts)
        split_checks.append(dict(partition=partition,global_counts=counts,
            class_coverage=manifest['class_coverage'],
            eligible_min2=[sum(row[c]>=2 for row in matrix) for c in range(10)],
            class_share_ess=[counts[c]**2/sum(row[c]**2 for row in matrix) for c in range(10)],
            idealized_local_balanced_mass=[sum(sum(row)/total/sum(v>0 for v in row)
                for row in matrix if row[c]>0) for c in range(10)],
            label_source='synthetic balanced 5000-per-class labels; not downloaded image data'))
    assert split_checks[0]['eligible_min2'][4] == split_checks[0]['eligible_min2'][8] == 0
    return dict(logit_shift=gauge, batch_balance=balanced, finite_batch_balance=finite_balanced,
                sampling_design=sampling_design,sampling=missing,split_checks=split_checks,
                control_variate=varp, absent_row_mask=negative,
                singleton_weight_cancellation="normalize((r*p)/r) == normalize(p), for every r>0",
                proto_ce_lower_bounds=[dict(classes=c,tau=0.5,
                    unrestricted_cosine=math.log1p((c-1)*math.exp(-4)),
                    nonnegative_relu=math.log1p((c-1)*math.exp(-2))) for c in [10,100,200]])


def audit():
    metric_keys = ['server_test_acc','shadow_fedavg_head_minus_actual_acc',
        'prototype_coverage','prototype_current_round_coverage','prototype_contributors_mean',
        'prototype_interclass_cosine_mean','prototype_same_class_client_cosine_mean',
        'prototype_server_centroid_cosine_mean','prototype_server_centroid_margin_mean',
        'prototype_nearest_class_acc','diag_aux_to_ce_loss_ratio','diag_grad_aux_to_ce_ratio',
        'diag_grad_ce_aux_cosine','head_effective_clients','sampled_class_coverage',
        'prototype_singleton_update_count','prototype_singleton_effective_momentum_mean']
    bpc = []
    for path in sorted((ROOT/'result').glob('*.csv')):
        data = rows(path)
        metrics = {key:s for key in metric_keys if (s:=summary(data,key)) is not None}
        pair_columns = [k for k in data[0] if k.startswith('diag_sampled_class_') and k.endswith('_samples')]
        pairs = [(float(r['sampled_class_coverage']),float(r['server_test_acc']))
                 for r in data[100:] if r.get('sampled_class_coverage') and r.get('server_test_acc')]
        bpc.append(dict(file=str(path.relative_to(ROOT)),rows=len(data),columns=len(data[0]),metrics=metrics,
                        sampling_fingerprint=hashlib.sha256(json.dumps(
                            [[r.get(k) for k in sorted(pair_columns)] for r in data]).encode()).hexdigest() if pair_columns else None,
                        last100_accuracy_coverage_pearson=pearson(pairs),
                        nonzero_aux_gradient_observations=sum(float(r.get('diag_grad_aux_norm') or 0)>0 for r in data),
                        negative_cosine_observations=sum(float(r.get('diag_grad_ce_aux_cosine') or 0)<0 for r in data),
                        transfer_columns=[k for k in data[0] if k.startswith('transfer/')]))
    btr = []
    for path in sorted((ROOT/'fedbtr_run_metrics').glob('*/experiment.json')):
        exp=json.loads(path.read_text()); final=json.loads((path.parent/'final.json').read_text())
        cfg, runtime = exp['experiment'], exp['runtime']
        metrics = final['last_global_test']; perclass=metrics['class_accuracy']; groups=runtime['groups']
        assert abs(statistics.mean(perclass)-metrics['macro_accuracy']) < 1e-10
        for group, indices in groups.items():
            assert abs(statistics.mean(perclass[c] for c in indices)-metrics[group]) < 1e-10
        btr.append(dict(directory=str(path.parent.relative_to(ROOT)),name=cfg.get('swanlab_setups', {}).get('experiment_name') or cfg.get('wandb_setups', {}).get('name'),
                        train_seed=cfg['train_setups']['seed'],split_sha256=runtime['split_sha256'],
                        data=cfg['data_setups'],train=cfg['train_setups'],
                        groups=groups,global_counts=runtime['global_class_counts'],
                        final_test=metrics,worst_class=min(perclass),best_round=final['best_round'],
                        validation_selected_test=final['best_validation_selected_test']))
    history=[]
    for path in sorted((ROOT/'fedbtr_run_metrics').glob('*.csv')):
        data=rows(path)
        pairs=[float(r['teacher_validation/macro_accuracy'])-float(r['global_before_validation/macro_accuracy'])
               for r in data if r.get('teacher_validation/macro_accuracy') and r.get('global_before_validation/macro_accuracy')]
        history.append(dict(file=str(path.relative_to(ROOT)),rows=len(data),
            teacher_gain_mean=statistics.mean(pairs) if pairs else None,teacher_gain_n=len(pairs),
            missing_delta=summary(data,'local_diagnostics/slot_000/missing_mean_delta'),
            last_validation=summary(data,'global_validation/macro_accuracy'),
            transfer_columns=[k for k in data[0] if k.startswith('transfer/')]))
    inventory=[]
    for folder in ['src','scripts','tests','docs','result','fedbtr_run_metrics']:
        for path in sorted((ROOT/folder).rglob('*')):
            if path.is_file() and path.suffix in ['.py','.json','.md','.csv','.txt']:
                inventory.append(dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,
                                      sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    output=dict(python=sys.version,root=str(ROOT),limitations=[
        "Historical FedBPC CSVs lack saved run configurations and original commit IDs.",
        "Numerical examples reproduce formulas with the standard library, not PyTorch execution.",
        "Rounds from one training trajectory are not independent replications.",
        "No training, dataset downloads, or changes to algorithms."],
        fedbpc=bpc,fedbtr=btr,fedbtr_history=history,synthetic=synthetic_diagnostics())
    (HERE/'audit_results.json').write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (HERE/'source_inventory.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2)+'\n')
    md=['# 历史日志复算表','', '自动生成；准确率单位 %，均为每个 CSV 实际最后一条记录。峰值不用于选模。',
        '','## FedBPC / FedAvg / FedProto','',
        '| 文件 | 轮数 | 最终 | 后20轮均值 | 后20轮标准差 |', '|---|---:|---:|---:|---:|']
    for run in bpc:
        s=run['metrics'].get('server_test_acc')
        if s: md.append(f"| {Path(run['file']).name} | {run['rows']} | {100*s['last']:.2f} | {100*s['last20_mean']:.3f} | {100*s['last20_sd']:.3f} |")
    md+=['','FedProto 是不同预测对象（客户端原型分类器的平均）；本表不得直接充当算法排名。FedBPC 文件名不证明配置。',
         '', '## FedBTR 固定最终轮测试','', '| 运行名 | seed | 划分指纹前12位 | Macro | Head | Middle | Tail | Worst |', '|---|---:|---|---:|---:|---:|---:|---:|']
    for run in btr:
        s=run['final_test']
        md.append('| '+run['name']+f" | {run['train_seed']} | {run['split_sha256'][:12]} | "+' | '.join(f'{100*s[k]:.2f}' for k in ['macro_accuracy','head','middle','tail'])+f" | {100*run['worst_class']:.2f} |")
    (HERE/'LOG_TABLES.md').write_text('\n'.join(md)+'\n')
    print(json.dumps(dict(fedbpc_csvs=len(bpc),fedbtr_runs=len(btr),fedbtr_csvs=len(history),
                         inventoried_files=len(inventory),outputs=['audit_results.json','source_inventory.json','LOG_TABLES.md']),ensure_ascii=False))


if __name__ == '__main__':
    audit()
