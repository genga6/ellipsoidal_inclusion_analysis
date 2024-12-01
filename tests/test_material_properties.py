import pytest
from core.material_properties import MaterialProperties

def test_material_properties():
    # 条件設定: ヤング率とポアソン比
    young_modulus = 70e9
    poisson_ratio = 0.33

    # MaterialPropertiesインスタンス生成
    material = MaterialProperties(young_modulus, poisson_ratio)

    # テスト: ヤング率とポアソン比が正しく設定されているか
    assert material.young_modulus == young_modulus
    assert material.poisson_ratio == poisson_ratio

    # テスト: シア弾性率
    expected_shear_modulus = young_modulus / (2 * (1 + poisson_ratio))
    assert material.shear_modulus == pytest.approx(expected_shear_modulus, rel=1e-3)

    # テスト: 体積弾性率
    expected_bulk_modulus = young_modulus / (3 * (1 - 2 * poisson_ratio))
    assert material.bulk_modulus == pytest.approx(expected_bulk_modulus, rel=1e-3)

    # テスト: ラメ定数（λ）
    expected_lambda_lame = (young_modulus * poisson_ratio) / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
    assert material.lambda_lame == pytest.approx(expected_lambda_lame, rel=1e-3)
