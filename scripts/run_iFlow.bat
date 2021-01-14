

for /l %%s in (61, 1, 100) do (
    python main.py ^
        -x 1000_40_5_5_3_%%s_gauss_xtanh_u_f ^
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
)