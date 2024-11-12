# Benchmark FSDP DLOJ-JZ
Benchmark GPU dense Computing -> ~ 98% GPU time

* Llama3.2-3B
* 50 training steps
* with FSDP
* with torch.compile

## Prerequisites
* Download Model : [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
* Download Dataset : [hieunguyenminh/roleplay](https://huggingface.co/datasets/hieunguyenminh/roleplay)

## Environment & Running
### On Jean-Zay
* Please use the `pytorch-gpu/py3/2.5.0` module
* Run benchmark with command:
  * `sbatch slurm/bench_h100_cap.slurm`
  * `sbatch slurm/bench_h100_nocap.slurm`

### Other system
(Please see `requierements.txt` to have module equivalence)

But used libraries list should be:
* torch==2.5.0
* transformers==4.46.0
* datasets==3.0.2
* idr_torch==2.2.0
* torchmetrics==1.5.1


