def test_imports():
    import low_bit_fake_quant as lbfq

    assert lbfq.QuantConfig().qk_quant == "fp8_block"
