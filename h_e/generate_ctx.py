import time

import tenseal as ts

if __name__ == '__main__':

    # context = ts.context(
    #     ts.SCHEME_TYPE.CKKS,
    #     poly_modulus_degree=8192,
    #     coeff_mod_bit_sizes=[60, 40, 40, 60]
    # )
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=16384,
        coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 40, 40, 40, 60]
    )

    context.global_scale = 2 ** 40
    context.generate_galois_keys()

    ctx_file = "ts_ckks.config"
    context_bytes = context.serialize(save_secret_key=True)
    f = open(ctx_file, "wb")
    f.write(context_bytes)

    pk_file = "ts_ckks_pk.config"
    pk_bytes = context.serialize(save_secret_key=False)
    f = open(pk_file, "wb")
    f.write(pk_bytes)
