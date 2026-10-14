# Simple import test for backbones modules

def test_import_backbones():
    import moviad.backbones
    import moviad.backbones.mcunet
    import moviad.backbones.micronet
    from micromind.networks.phinet import PhiNet
    assert callable(PhiNet)
