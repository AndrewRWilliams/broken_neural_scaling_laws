# Code Release for "Broken Neural Scaling Laws" (BNSL) paper ([arxiv.org/abs/2210.14891](https://arxiv.org/abs/2210.14891))

Read Appendix A.6 of arXiv version of this paper for more details on what this code does.

To reproduce the Fitting and Extrapolation of BNSL on 4 Digit Addition from Figure 4 Left, run 

```python fit_bnsl_and_extrapolate__4_digit_addition__dataset_size_x-axis.py```


To reproduce the Fitting and Extrapolation of BNSL on a noiseless simulation of the scaling behavior of 4 Digit Addition from Figure 4 Right, run 

```python fit_bnsl_and_extrapolate__4_digit_addition__dataset_size_x-axis__noiseless_simulation.py```




To reproduce the Decomposition of BNSL into Power Law Segments from Figure 1, run 

```python make_figure_1__decomposition_of_bnsl_into_power_law_segments.py ```


# Note:

🚨🚨🚨

**When you fit a BNSL to your own scaling data, you may need to adjust the grid search range and resolution to get a good fit.**

🚨🚨🚨

# Here is some bibtex to use for citation: 

```
@inproceedings{
caballero2023broken,
title={Broken Neural Scaling Laws},
author={Ethan Caballero and Kshitij Gupta and Irina Rish and David Krueger},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://arxiv.org/abs/2210.14891}
}
```
