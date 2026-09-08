import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

from apps.utils import read_csv_with_fallback, select_or_upload_dataset


CLASSIFICATION_MODELS = {
    "逻辑回归": lambda: LogisticRegression(max_iter=1000, random_state=1234),
    "XGBoost": lambda: XGBClassifier(
        eval_metric="logloss",
        n_estimators=100,
        random_state=1234,
    ),
    "决策树": lambda: DecisionTreeClassifier(random_state=1234),
    "随机森林": lambda: RandomForestClassifier(n_estimators=200, random_state=1234),
    "支持向量机": lambda: SVC(random_state=1234),
    "AdaBoost": lambda: AdaBoostClassifier(random_state=1234),
}

REGRESSION_MODELS = {
    "线性回归": lambda: LinearRegression(),
    "XGBoost": lambda: XGBRegressor(
        objective="reg:squarederror",
        n_estimators=100,
        random_state=1234,
    ),
    "决策树": lambda: DecisionTreeRegressor(random_state=1234),
    "随机森林": lambda: RandomForestRegressor(n_estimators=200, random_state=1234),
    "支持向量机": lambda: SVR(),
    "AdaBoost": lambda: AdaBoostRegressor(random_state=1234),
}


def make_one_hot_encoder():
    try:
        return OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=5,
            sparse_output=True,
        )
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def build_preprocessor(X):
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.columns.difference(numeric_features).tolist()

    transformers = []
    if numeric_features:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )

    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_one_hot_encoder()),
                    ]
                ),
                categorical_features,
            )
        )

    return ColumnTransformer(transformers=transformers)


def can_stratify(y):
    class_counts = pd.Series(y).value_counts()
    return len(class_counts) > 1 and class_counts.min() >= 2


def infer_task_type(y):
    if pd.api.types.is_numeric_dtype(y) and y.dropna().nunique() > 25:
        return "回归"
    return "分类"


def detect_identifier_columns(X):
    id_keywords = ["id", "编号", "订单号", "序号", "编码", "手机号", "身份证", "sku"]
    row_count = max(len(X), 1)
    identifier_columns = []
    for column in X.columns:
        column_name = str(column).lower()
        unique_ratio = X[column].nunique(dropna=True) / row_count
        name_looks_like_id = any(keyword in column_name for keyword in id_keywords)
        value_looks_unique = (
            unique_ratio >= 0.95
            and not pd.api.types.is_numeric_dtype(X[column])
            and not pd.api.types.is_bool_dtype(X[column])
        )
        if name_looks_like_id or value_looks_unique:
            identifier_columns.append(column)
    return identifier_columns


def make_model_signature(dataset_hash, target_column, task_type, selected_algo, k, test_size, auto_drop_id_columns):
    return {
        "dataset_hash": dataset_hash,
        "target_column": target_column,
        "task_type": task_type,
        "selected_algo": selected_algo,
        "k": k,
        "test_size": test_size,
        "auto_drop_id_columns": auto_drop_id_columns,
    }


def clear_trained_model():
    for key in ["ml_pipeline", "model_info", "label_encoder", "feature_columns", "task_type", "model_signature"]:
        st.session_state[key] = None


def invalidate_model_if_needed(signature):
    current_signature = st.session_state.get("model_signature")
    if st.session_state.get("ml_pipeline") is not None and current_signature != signature:
        clear_trained_model()
        st.info("训练配置已变化，旧模型已失效。请重新训练后再预测。")


def build_pipeline(X, selected_algo, k, task_type):
    score_func = f_regression if task_type == "回归" else f_classif
    model_factory = REGRESSION_MODELS if task_type == "回归" else CLASSIFICATION_MODELS
    steps = [
        ("preprocess", build_preprocessor(X)),
        ("feature_selection", SelectKBest(score_func=score_func, k=k)),
        ("model", model_factory[selected_algo]()),
    ]
    return Pipeline(steps=steps)


def show_dataset_preview(df):
    if st.checkbox("查看数据集"):
        number = st.number_input("查看行数", min_value=1, value=5)
        st.dataframe(df.head(number))


def show_target_summary(df, target_column):
    if st.checkbox("查看目标字段取值计数"):
        st.write(df[target_column].value_counts())


def evaluate_classifier(pipeline, X_test, y_test, label_encoder):
    y_pred = pipeline.predict(X_test)
    labels = label_encoder.classes_

    metrics_df = pd.DataFrame(
        [
            ["准确率", accuracy_score(y_test, y_pred)],
            ["精确率（加权）", precision_score(y_test, y_pred, average="weighted", zero_division=0)],
            ["召回率（加权）", recall_score(y_test, y_pred, average="weighted", zero_division=0)],
            ["F1 值（加权）", f1_score(y_test, y_pred, average="weighted", zero_division=0)],
        ],
        columns=["指标", "数值"],
    )

    if len(labels) == 2 and hasattr(pipeline, "predict_proba"):
        y_score = pipeline.predict_proba(X_test)[:, 1]
        metrics_df.loc[len(metrics_df)] = ["ROC-AUC", roc_auc_score(y_test, y_score)]

    st.subheader("评估指标")
    st.table(metrics_df)

    st.subheader("混淆矩阵")
    matrix = confusion_matrix(y_test, y_pred)
    st.dataframe(
        pd.DataFrame(matrix, index=labels, columns=labels),
    )

    st.subheader("分类报告")
    report = classification_report(
        y_test,
        y_pred,
        target_names=[str(label) for label in labels],
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose()
    report_df.rename(
        columns={
            "precision": "精确率",
            "recall": "召回率",
            "f1-score": "F1 值",
            "support": "样本数",
        },
        index={
            "accuracy": "准确率",
            "macro avg": "宏平均",
            "weighted avg": "加权平均",
        },
        inplace=True,
    )
    st.dataframe(report_df)


def evaluate_regressor(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    metrics_df = pd.DataFrame(
        [
            ["平均绝对误差 MAE", mean_absolute_error(y_test, y_pred)],
            ["均方误差 MSE", mean_squared_error(y_test, y_pred)],
            ["均方根误差 RMSE", mean_squared_error(y_test, y_pred) ** 0.5],
            ["决定系数 R²", r2_score(y_test, y_pred)],
        ],
        columns=["指标", "数值"],
    )
    st.subheader("评估指标")
    st.table(metrics_df)

    comparison_df = pd.DataFrame({"真实值": y_test, "预测值": y_pred})
    st.subheader("预测对比")
    st.dataframe(comparison_df.head(200))


def show_cv_score(pipeline, X_train, y_train, task_type):
    sample_count = len(y_train)
    if sample_count < 5:
        st.caption("训练样本少于 5 行，已跳过交叉验证。")
        return

    if task_type == "分类":
        min_class_count = pd.Series(y_train).value_counts().min()
        cv = min(5, int(min_class_count))
        if cv < 2:
            st.caption("部分类别样本过少，已跳过交叉验证。")
            return
        scoring = "f1_weighted"
        metric_name = "交叉验证 F1（加权）"
    else:
        cv = min(5, sample_count)
        scoring = "r2"
        metric_name = "交叉验证 R²"

    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scoring)
    st.metric(metric_name, f"{scores.mean():.4f}", delta=f"标准差 {scores.std():.4f}")


def predict_uploaded_file(uploaded_file):
    pipeline = st.session_state.get("ml_pipeline")
    label_encoder = st.session_state.get("label_encoder")
    feature_columns = st.session_state.get("feature_columns")
    task_type = st.session_state.get("task_type")

    if pipeline is None or feature_columns is None or task_type is None:
        st.warning("请先训练模型，再上传数据进行预测。")
        return

    prediction_df = read_csv_with_fallback(uploaded_file)
    missing_columns = [column for column in feature_columns if column not in prediction_df.columns]
    if missing_columns:
        st.error(f"预测文件缺少必要字段：{missing_columns}")
        return

    X_new = prediction_df[feature_columns]
    predictions = pipeline.predict(X_new)
    if task_type == "分类":
        prediction_df["预测结果"] = label_encoder.inverse_transform(predictions)
    else:
        prediction_df["预测结果"] = predictions
    st.dataframe(prediction_df)


def modelling(df):
    st.session_state.setdefault("ml_pipeline", None)
    st.session_state.setdefault("model_info", None)
    st.session_state.setdefault("label_encoder", None)
    st.session_state.setdefault("feature_columns", None)
    st.session_state.setdefault("task_type", None)
    st.session_state.setdefault("model_signature", None)

    show_dataset_preview(df)

    target_column = st.selectbox("选择目标字段", df.columns)

    original_rows = len(df)
    df = df.dropna(subset=[target_column]).copy()
    dropped_rows = original_rows - len(df)
    if dropped_rows:
        st.warning(f"已删除 {dropped_rows} 行目标字段为空的数据。")

    show_target_summary(df, target_column)

    X = df.drop(columns=[target_column])
    y_raw = df[target_column]
    auto_drop_id_columns = st.checkbox(
        "自动排除疑似 ID 字段",
        value=True,
        help="编号、订单号、手机号、SKU 或几乎每行都不同的字段通常不适合直接 One-Hot。",
    )
    if auto_drop_id_columns:
        identifier_columns = detect_identifier_columns(X)
        if identifier_columns:
            X = X.drop(columns=identifier_columns)
            st.info(f"已排除疑似 ID 字段：{identifier_columns}")

    inferred_task_type = infer_task_type(y_raw)
    task_type = st.selectbox(
        "选择任务类型",
        ["分类", "回归"],
        index=0 if inferred_task_type == "分类" else 1,
        help="类别型目标适合分类；连续数值目标适合回归。",
    )
    model_options = CLASSIFICATION_MODELS if task_type == "分类" else REGRESSION_MODELS
    selected_algo = st.selectbox("选择算法", list(model_options.keys()))

    if X.empty:
        st.error("请选择至少包含一个特征字段的数据集。")
        return

    if task_type == "分类":
        unique_targets = y_raw.dropna().nunique()
        if unique_targets < 2:
            st.error("分类目标字段至少需要包含两个类别。")
            return
        if unique_targets > 25:
            st.warning("该字段类别数量较多，更像连续数值目标；建议切换为回归任务。")
    else:
        y_raw = pd.to_numeric(y_raw, errors="coerce")
        valid_rows = y_raw.notna()
        dropped_regression_rows = len(y_raw) - int(valid_rows.sum())
        if dropped_regression_rows:
            st.warning(f"已删除 {dropped_regression_rows} 行无法转换为数值的目标数据。")
        X = X.loc[valid_rows]
        y_raw = y_raw.loc[valid_rows]
        if len(y_raw) < 3:
            st.error("回归任务至少需要 3 行有效数值目标数据。")
            return

    use_feature_selection = st.checkbox(
        "手动选择 Top K 特征",
        value=False,
        help="默认使用全部特征；One-Hot 后的实际特征数可能多于原始字段数。",
    )
    if use_feature_selection:
        k_options = list(range(1, min(20, X.shape[1]) + 1))
        k = st.selectbox("选择 Top K 特征", k_options)
    else:
        k = "all"

    test_size = st.slider(
        "选择测试集比例",
        min_value=0.1,
        max_value=0.5,
        value=0.2,
        step=0.05,
        help="当每个目标类别至少有两个样本时，会自动使用分层抽样。",
    )
    model_signature = make_model_signature(
        st.session_state.get("dataset_hash"),
        target_column,
        task_type,
        selected_algo,
        k,
        test_size,
        auto_drop_id_columns,
    )
    invalidate_model_if_needed(model_signature)

    if st.button("开始训练", help="使用防止数据泄漏的 Pipeline 训练所选分类器"):
        label_encoder = None
        if task_type == "分类":
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y_raw.astype(str))
            stratify = y if can_stratify(y) else None
        else:
            y = y_raw.astype(float)
            stratify = None

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=1234,
                stratify=stratify,
            )

            pipeline = build_pipeline(X_train, selected_algo, k, task_type)
            pipeline.fit(X_train, y_train)
        except ValueError as exc:
            st.error(f"训练失败：{exc}")
            return

        st.session_state["ml_pipeline"] = pipeline
        st.session_state["label_encoder"] = label_encoder
        st.session_state["feature_columns"] = X.columns.tolist()
        st.session_state["task_type"] = task_type
        st.session_state["model_signature"] = model_signature
        st.session_state["model_info"] = f"{selected_algo} 已完成{task_type}任务训练，使用防止数据泄漏的预处理 Pipeline。"

        st.success(st.session_state["model_info"])
        show_cv_score(pipeline, X_train, y_train, task_type)
        if task_type == "分类":
            evaluate_classifier(pipeline, X_test, y_test, label_encoder)
        else:
            evaluate_regressor(pipeline, X_test, y_test)

    st.subheader("上传测试数据进行预测")
    uploaded_file = st.file_uploader("选择 CSV 文件", type="csv")
    if uploaded_file is not None and st.button("开始预测"):
        predict_uploaded_file(uploaded_file)


def app():
    st.title("机器学习建模")
    st.subheader("选择数据集")
    select_or_upload_dataset(modelling)


if __name__ == "__main__":
    app()
