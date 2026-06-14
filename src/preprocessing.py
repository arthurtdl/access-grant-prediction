from pathlib import Path

import joblib
import numpy as np
import scipy.sparse as sp
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

RANDOM_STATE = 42


def split_data(X, y, test_size=0.20, val_size=0.25, random_state=RANDOM_STATE):
    """Separa os dados em treino, validação e teste interno de forma estratificada.

    O `test_size` define o percentual do total destinado ao teste interno.
    O `val_size` define o percentual do restante (treino + validação)
    destinado à validação.
    """
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor(colunas_categoricas, min_frequency=10):
    """Cria o ColumnTransformer com OneHotEncoder para os atributos categóricos.

    Categorias com menos de `min_frequency` ocorrências no treino são agrupadas
    em uma categoria "infrequente", reduzindo a dimensionalidade gerada pelos
    atributos de alta cardinalidade (ex.: RESOURCE, MGR_ID).
    """
    return ColumnTransformer(
        transformers=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_frequency),
                colunas_categoricas,
            )
        ]
    )


def apply_smote(X_train, y_train, random_state=RANDOM_STATE, sampling_strategy="auto"):
    """Aplica SMOTE para balancear as classes do conjunto de treino.

    Deve ser usado SOMENTE no conjunto de treino, e somente após o
    pré-processamento (ex.: One-Hot Encoding). Validação e teste mantêm a
    distribuição original para que a avaliação reflita o cenário real.
    """
    smote = SMOTE(random_state=random_state, sampling_strategy=sampling_strategy)
    return smote.fit_resample(X_train, y_train)


def save_processed_data(output_dir, preprocessador, X_train_res, y_train_res, X_val, y_val, X_test, y_test):
    """Salva os artefatos de pré-processamento para a etapa de modelagem.

    Cada integrante da equipe pode carregar esses arquivos com
    `load_processed_data` para treinar e avaliar seu modelo sem repetir o
    pré-processamento.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sp.save_npz(output_dir / "X_train_res.npz", sp.csr_matrix(X_train_res))
    sp.save_npz(output_dir / "X_val.npz", sp.csr_matrix(X_val))
    sp.save_npz(output_dir / "X_test.npz", sp.csr_matrix(X_test))

    np.save(output_dir / "y_train_res.npy", np.asarray(y_train_res))
    np.save(output_dir / "y_val.npy", np.asarray(y_val))
    np.save(output_dir / "y_test.npy", np.asarray(y_test))

    joblib.dump(preprocessador, output_dir / "preprocessador.pkl")


def load_processed_data(input_dir):
    """Carrega os artefatos salvos por `save_processed_data`."""
    input_dir = Path(input_dir)

    X_train_res = sp.load_npz(input_dir / "X_train_res.npz")
    X_val = sp.load_npz(input_dir / "X_val.npz")
    X_test = sp.load_npz(input_dir / "X_test.npz")

    y_train_res = np.load(input_dir / "y_train_res.npy")
    y_val = np.load(input_dir / "y_val.npy")
    y_test = np.load(input_dir / "y_test.npy")

    preprocessador = joblib.load(input_dir / "preprocessador.pkl")

    return X_train_res, X_val, X_test, y_train_res, y_val, y_test, preprocessador
