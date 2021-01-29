@echo off
setlocal enabledelayedexpansion

del "results\cc_across_dims\iFlow_1000_40_5.txt"

set s=1
for /D %%F in ("experiments\*") do (
    python calculate_CC_per_dimension.py ^
        -x 1000_40_5_5_3_!s!_gauss_xtanh_u_f ^
        -ml %%F ^
        -i iFlow ^
        -ft RQNSF_AG ^
        -npa Softplus ^
        -fl 10 ^
        -lr_df 0.25 ^
        -lr_pn 10 ^
        -b 64 ^
        -e 20 ^
        -l 1e-3 ^
        -s 1 ^
        -u 0 ^
        -c ^
        -p

    set /a s=s+1
)
