![intellifold Cover](assets/intellifold-cover.png)

# IntelliFold: A Controllable Foundation Model for General and Specialized Biomolecular Structure Prediction
[![HuggingFace Models](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Models-yellow)](https://huggingface.co/intelligenAI/intellifold)
[![PyPI](https://img.shields.io/pypi/v/intellifold)](https://pypi.org/project/intellifold/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Email](https://img.shields.io/badge/Email-Contact-lightgrey?logo=gmail)](#contact-us)


<div align="center" style="margin: 20px 0;">
  <span style="margin: 0 10px;">⚡ <a href="https://server.intfold.com">IntelliFold Server</a></span>
  &bull; <span style="margin: 0 10px;">📄 <a href="https://www.biorxiv.org/content/10.64898/2026.02.09.704787v1">IntelliFold 2 Release Note</a></span> &bull; <span style="margin: 0 10px;">📄 <a href="https://arxiv.org/abs/2507.02025">IntelliFold Technical Report</a></span>
</div>


![IntelliFold Model](assets/Intellifold-Model-Arc.png)

## 🎉 New Model Release

 - **2026-02-07**: We are excited to present [[IntelliFold 2]](https://www.biorxiv.org/content/10.64898/2026.02.09.704787v1). This version represents a
major architectural update and is one of the first open-source models to outperform AlphaFold3 on
Foldbench.  


## 📊 Benchmarking
To comprehensively evaluate the performance of IntelliFold 2, we conducted a rigorous evaluation on [FoldBench](https://github.com/BEAM-Labs/FoldBench). We compared IntelliFold against several leading methods, including [Boltz-1,2](https://github.com/jwohlwend/boltz), [Chai-1](https://github.com/chaidiscovery/chai-lab), [Protenix](https://github.com/bytedance/Protenix) and [Alphafold3](https://github.com/google-deepmind/alphafold3).

For more details on the benchmarking process and results, please refer to our release note [IntelliFold 2 Release Note](https://www.biorxiv.org/content/10.64898/2026.02.09.704787v1) and [IntelliFold Technical Report](https://arxiv.org/abs/2507.02025).

![Benchmark Metrics](assets/Intellifold_v2_performance.png)


## 🚀 Quick Start

To quickly get started with IntelliFold, you can use the following commands:
>**model**: Choose the model for inference via `--model`. Default: `v2-flash`. Supported: `v1`, `v2`, `v2-flash`.  
>For differences between model versions, please refer to our **release note** [IntelliFold 2 Release Note](https://www.biorxiv.org/content/10.64898/2026.02.09.704787v1)

```bash
# Install intellifold from PyPI
pip install intellifold
# Run inference with an example YAML file, using the default model (v2-Flash)
intellifold predict ./examples/5S8I_A.yaml --out_dir ./output --cache ./cache_data --model v2-flash
```

## ⚙️ Installation

To more complete installation instructions and usage, please refer to the [Installation Guide](docs/installation.md).


## 🔍 Inference

1. **Prepare Input File**: Create a YAML file with your sequences following our [input format specification](docs/input_yaml_format.md)
      > **Note**: **Our model natively supports template inputs**, but this repo’s inference pipeline (adapted from Boltz-1) does not currently include template processing. Predictions therefore run **without templates**, which may slightly reduce performance.

2. **Run Prediction**:
   ```bash
   intellifold predict your_input.yaml --out_dir ./results
   ```

    IntelliFold v2-Flash will be used by default, you can also use IntelliFold v2 by specifying the model name:
    ```bash
    intellifold predict your_input.yaml --out_dir ./results --model v2
    ```

3. **Check Results**: Find predicted structures and confidence scores in the output directory, you can also check the section of **output format** in [output documentation](docs/input_yaml_format.md#output-format).

4. **Optional Optimization**: Enable [custom kernels](docs/kernels.md) for faster inference and reduced memory usage

5. **Custom MSA Servers**: For enterprise or private deployments requiring authentication, see the [MSA Server Authentication](docs/usage.md#msa-server-authentication) section in our usage guide.

For comprehensive usage instructions and examples, refer to the [Usage Guide](docs/usage.md).


## 🌐 IntelliFold Server

**We highly recommend using the [IntelliFold Server](https://server.intfold.com) for the most accurate, complete, and convenient biomolecular structure predictions.** It requires no installation and provides an intuitive web interface to submit your sequences and visualize results directly in your browser. The server runs the **full, optimized, latest** IntelliFold implementation for optimal performance.

![IntelliFold Server](assets/intellifold-server-screenshot.png)


## 📜 Citation

If you use IntelliFold in your research, please cite our paper:

```
@article{qiao2026intellifold,
  title={IntelliFold-2: Surpassing AlphaFold 3 via Architectural Refinement and Structural Consistency},
  author={Qiao, Lifeng and Yan, He and Liu, Gary and Guo, Gaoxing and Sun, Siqi},
  journal={bioRxiv},
  year={2026},
  publisher={Cold Spring Harbor Laboratory}
}

@article{team2025intfold,
  title={IntFold: A Controllable Foundation Model for General and Specialized Biomolecular Structure Prediction},
  author={Team, The IntFold and Qiao, Leon and Bai, Wayne and Yan, He and Liu, Gary and Xi, Nova and Zhang, Xiang and Sun, Siqi},
  journal={arXiv preprint arXiv:2507.02025},
  year={2025}
}
```


## 🔗 Acknowledgements

- The implementation of **fast layernorm operators** is inspired by [OneFlow](https://github.com/Oneflow-Inc/oneflow) and [FastFold](https://github.com/hpcaitech/FastFold), following [Protenix](https://github.com/bytedance/Protenix)'s usage. 
- Many components in `intellifold/openfold/` are adapted from [OpenFold](https://github.com/aqlaboratory/openfold), with substantial modifications and improvements by our team (except for the `LayerNorm` part).  
- This repository, the implementation of **Inference Data Pipeline**(Data/Feature Processing and MSA generation tasks) referred to [Boltz-1](https://github.com/jwohlwend/boltz), and modify some codes to adapt to the input of our model.



## ⚖️ License

The IntelliFold project, including code and model parameters, is made available under the [Apache 2.0 License](./LICENSE), it is free for both academic research and commercial use.

## 📬 Contact Us

If you have any questions or are interested in collaboration, please feel free to contact us at contact@intfold.com.

