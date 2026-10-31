"""Model factory. One instance per run is refit for every coalition: TabPFN/TabICL `fit`
only stores the context, and reusing the instance avoids reloading checkpoints."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

GPU_MODELS = {"tabpfn_v2", "tabpfn_v2_5", "tabicl_v2"}


def make_model(name: str, seed: int, *, device: str = "cuda", n_estimators: int = 4,
               cpu_jobs: int = 4):
    if name in ("tabpfn_v2", "tabpfn_v2_5"):
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion

        version = ModelVersion.V2 if name == "tabpfn_v2" else ModelVersion.V2_5
        return TabPFNClassifier.create_default_for_version(
            version, device=device, n_estimators=n_estimators, random_state=seed
        )
    if name == "tabicl_v2":
        from tabicl import TabICLClassifier

        return TabICLClassifier(device=device, n_estimators=n_estimators, random_state=seed)
    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.8,
                             tree_method="hist", n_jobs=cpu_jobs, random_state=seed)
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, n_jobs=cpu_jobs, random_state=seed)
    if name == "logreg":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=1.0, max_iter=2000))
    raise ValueError(f"unknown model: {name}")
