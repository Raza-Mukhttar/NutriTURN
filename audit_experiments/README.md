# audit_experiments

One script per experiment in the paper, named for what it does. 51 experiment scripts.

```bash
export PYTHONHASHSEED=0
bash audit_experiments/run_all_experiments.sh      # everything, in dependency order, a few hours
python audit_experiments/stationary_baselines.py   # or any single experiment
```


