import pandas as pd

from triage.data.build_dataset import asignar_clase, leer_crudos, limpiar, muestrear

LABEL_MAP = {
    "clases": {"tarjetas": ["Credit card"], "cobranza": ["Debt collection"]},
    "clase_por_defecto": "otros",
    "por_subproducto": {},
}


def datos(**columnas) -> pd.DataFrame:
    base = {"complaint_id": ["3", "1", "2"], "fecha": ["2023-01-03", "2023-01-01", "2023-01-02"],
            "producto": ["Credit card", "Debt collection", "Mortgage"],
            "subproducto": [None, None, None], "texto": ["a", "b", "c"]}
    base.update(columnas)
    return pd.DataFrame(base)


def test_limpiar_quita_vacios_y_ordena():
    df = limpiar(datos(texto=["hola", "  ", "adios"]))
    assert list(df["complaint_id"]) == [2, 3]


def test_limpiar_quita_textos_duplicados():
    df = limpiar(datos(texto=["Mismo  texto", "mismo texto", "otro"]))
    assert len(df) == 2
    assert 1 in set(df["complaint_id"])  # se conserva el más antiguo


def test_asignar_clase_usa_mapa_y_clase_por_defecto():
    clases = asignar_clase(datos(), LABEL_MAP)
    assert list(clases) == ["tarjetas", "cobranza", "otros"]


def test_subproducto_tiene_prioridad():
    label_map = {**LABEL_MAP, "por_subproducto": {"Gift card": "prepago"}}
    df = datos(subproducto=["Gift card", None, None])
    assert asignar_clase(df, label_map).iloc[0] == "prepago"


def test_muestrear_respeta_tope_y_es_determinista():
    df = pd.DataFrame({"complaint_id": range(100), "clase": ["a"] * 80 + ["b"] * 20})
    m1 = muestrear(df, max_por_clase=10, seed=42)
    m2 = muestrear(df, max_por_clase=10, seed=42)
    assert m1["clase"].value_counts().max() == 10
    assert m1.equals(m2)


def crudo(ids: list[int], fecha: str) -> pd.DataFrame:
    """Reclamos con las columnas originales del CFPB."""
    return pd.DataFrame({
        "Date received": [fecha] * len(ids), "Product": ["Credit card"] * len(ids),
        "Sub-product": [None] * len(ids), "Issue": ["x"] * len(ids),
        "Consumer complaint narrative": [f"texto {i}" for i in ids], "Complaint ID": ids,
    })


def test_leer_crudos_une_csv_y_parquet_de_landing(tmp_path):
    raw, landing = tmp_path / "raw", tmp_path / "landing"
    raw.mkdir()
    landing.mkdir()
    crudo([1, 2], "2023-06-01").to_csv(raw / "extracto.csv", index=False)
    crudo([3], "2024-01-01").to_parquet(landing / "2024-01-01.parquet", index=False)
    sin_texto = crudo([4], "2024-01-02").assign(**{"Consumer complaint narrative": [None]})
    sin_texto.to_parquet(landing / "2024-01-02.parquet", index=False)

    df = leer_crudos(str(raw), str(landing))
    assert list(df.columns) == ["complaint_id", "fecha", "producto", "subproducto", "texto"]
    assert list(df["complaint_id"]) == ["1", "2", "3", "4"]
    assert list(limpiar(df)["complaint_id"]) == [1, 2, 3]  # el nulo del parquet se descarta


def test_leer_crudos_sin_landing_solo_lee_csv(tmp_path):
    crudo([1], "2023-06-01").to_csv(tmp_path / "extracto.csv", index=False)
    df = leer_crudos(str(tmp_path), str(tmp_path / "no_existe"))
    assert list(df["complaint_id"]) == ["1"]
