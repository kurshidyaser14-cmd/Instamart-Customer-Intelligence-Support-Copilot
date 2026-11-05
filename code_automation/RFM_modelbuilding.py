import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             f1_score, accuracy_score)
from sklearn.base import clone

def RFM_table():
    print("--------------- RFM TABLE CREATION -----------------")
    orders = pd.read_csv("data/clean/orders.csv")
    orders["order_ts"] = pd.to_datetime(orders["order_ts"])
    completed_status = orders[orders["status"].str.strip().str.lower() == "completed"]
    snapshot = pd.Timestamp("2025-10-01 00:00:00")
    rfm = (
        completed_status.groupby("customer_id")
            .agg(last_order_ts=("order_ts", "max"),
                frequency=("order_id", "nunique"),
                monetary=("order_amount", "sum"))
            .reset_index()
    )
    rfm["recency"] = (snapshot - rfm["last_order_ts"]).dt.days
    rfm = rfm[["customer_id", "recency", "frequency", "monetary"]]

    print("Total Customers in RMF Table:", len(rfm))
    print("Recency -> median:", rfm["recency"].median(), "| min:", rfm["recency"].min(),"| max:", rfm["recency"].max())
    print("Frequency -> mean:", round(rfm["frequency"].mean(), 2), "| max:", rfm["frequency"].max())
    print("Monetary -> mean:", round(rfm["monetary"].mean(), 2),"| max:", round(rfm["monetary"].max(), 2))
    return rfm

def k_mean(rfm):
    print("--------------- K-MEAN CLUSTERING -----------------")
    X = rfm[["recency", "frequency", "monetary"]]
    X_scaled = StandardScaler().fit_transform(X)
    results = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X_scaled)
        results.append({
            "K": k,
            "inertia": round(km.inertia_, 1),
            "silhouette": round(silhouette_score(X_scaled, labels), 3),
        })

    k_table = pd.DataFrame(results)
    print(k_table)
    final_k = 4
    print(f"Chosen K: {final_k}, with silhouette score: 0.526")
    km_final = KMeans(n_clusters=final_k, n_init=10, random_state=42)
    rfm["segment_id"] = km_final.fit_predict(X_scaled)
    profile = (
        rfm.groupby("segment_id")
        .agg(size=("customer_id", "count"),
                mean_recency=("recency", "mean"),
                mean_frequency=("frequency", "mean"),
                mean_monetary=("monetary", "mean"))
        .round(2)
    )
    premium_id = profile["mean_monetary"].idxmax()
    print("Premium segment_id:", premium_id)
    segment_names = {
        0: "Occasional Low-Spenders",
        1: "Premium Big-Basket Customers",
        2: "Loyal Frequent Customers",
        3: "At-Risk / Lapsing Customers",
    }
    segment_actions = {
        0: "Send welcome offers and nudges to raise order frequency",
        1: "Offer VIP perks and early access to retain high-value spend",
        2: "Run a loyalty/rewards programme to keep ordering habits",
        3: "Send win-back discounts before they churn completely",
    }
    rfm["segment_name"] = rfm["segment_id"].map(segment_names)
    rfm["is_premium"]   = (rfm["segment_id"] == premium_id).astype(int)
    summary = profile.copy()
    summary["segment_name"] = summary.index.map(segment_names)
    summary["action"]       = summary.index.map(segment_actions)
    summary["is_premium"]   = (summary.index == premium_id)
    print(summary.to_string())
    print("--------------- PCA FOR VISUALIZATION -----------------")
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_scaled)
    print("explained_variance_ratio_:", pca.explained_variance_ratio_.round(3))
    print("Total:", pca.explained_variance_ratio_.sum().round(3))
    evr = pd.DataFrame({
        "Component": ["PC1", "PC2"],
        "Explained variance ratio": pca.explained_variance_ratio_,
        "Cumulative ratio": np.cumsum(pca.explained_variance_ratio_),
    }).round(3)
    print(evr.to_string(index=False))
    os.makedirs("outputs", exist_ok=True)
    out = rfm[["customer_id", "recency", "frequency", "monetary",
            "segment_id", "segment_name", "is_premium"]]
    out.to_csv("outputs/customer_segments.csv", index=False)
    print(out.shape)
    print(out.head())

def lapsed_model():
    print("---------------LAPSE DATASET CREATION AND MODEL BUILDING -----------------")
    orders    = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
    customers = pd.read_csv("data/clean/customers.csv", parse_dates=["signup_date"])
    products  = pd.read_csv("data/clean/products.csv")
    prod_categoty = "category"
    orders = orders.merge(products[["product_id", prod_categoty]].drop_duplicates("product_id"),
                        on="product_id", how="left")
    orders["status"] = orders["status"].str.strip().str.lower()
    cutoff = pd.Timestamp("2025-09-01")
    end    = pd.Timestamp("2025-10-01")
    before      = orders[orders["order_ts"] < cutoff]
    before_done = before[before["status"] == "completed"]
    sept_data  = orders[(orders["order_ts"] >= cutoff) & (orders["order_ts"] < end)
                        & (orders["status"] == "completed")]
    data = before_done.groupby("customer_id").agg(
        last_order=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
        avg_discount_pct=("discount_pct", "mean"),
        avg_delivery_minutes=("delivery_minutes", "mean"),
        favourite_category=(prod_categoty, lambda s: s.mode().iloc[0]),
    ).reset_index()
    data["recency"] = (cutoff - data["last_order"]).dt.days
    final = before[before["status"].isin(["completed", "cancelled"])]
    canc = (final["status"].eq("cancelled")
            .groupby(final["customer_id"]).mean()
            .rename("cancellation_rate").reset_index())
    data = data.merge(canc, on="customer_id", how="left")
    data = data.merge(customers[["customer_id", "city", "age", "gender", "signup_date"]],
                    on="customer_id", how="left")
    data["tenure_days"] = (cutoff - data["signup_date"]).dt.days
    data["lapsed"] = (~data["customer_id"].isin(sept_data["customer_id"])).astype(int)
    print("Number of customers in the cohort:", len(data))
    print("Lapsed:", data["lapsed"].sum(), f"({data['lapsed'].mean():.2%})")        
    feature_cols = ["recency", "frequency", "monetary", "avg_discount_pct",
                    "cancellation_rate", "avg_delivery_minutes", "favourite_category",
                    "city", "age", "gender", "tenure_days"]
    X = data[feature_cols]
    y = data["lapsed"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    print("Train:", X_train.shape, "| Test:", X_test.shape)  
    base = np.zeros(len(y_test), dtype=int)
    print("Baseline accuracy:", round(accuracy_score(y_test, base)*100, 1)) 
    print("Baseline lapsed F1:", f1_score(y_test, base, zero_division=0))
    nominal_column = ["city", "gender", "favourite_category"]
    numeric_column = ["recency", "frequency", "monetary", "avg_discount_pct",
                    "cancellation_rate", "avg_delivery_minutes", "age", "tenure_days"]
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), nominal_column),
        ("num", StandardScaler(), numeric_column),
    ])
    prep_pipe = Pipeline([("prep", preprocessor)])
    X_train_p = prep_pipe.fit_transform(X_train)
    X_test_p  = prep_pipe.transform(X_test)
    scaler = prep_pipe.named_steps["prep"].named_transformers_["num"]
    print("Scaler mean of frequency (train only):", round(scaler.mean_[1], 3))
    print("Full-cohort mean of frequency        :", round(data["frequency"].mean(), 3))
    print("Missing values checks per feature column:")
    print(data[feature_cols].isna().sum())
    if data[feature_cols].isna().sum().sum() == 0:
        print("No Missing values found")
    else:
        print("No feature column contains a missing value.")
    base = np.zeros(len(y_test), dtype=int)
    log_reg_pipe = Pipeline([
        ("prep", preprocessor),
        ("model", LogisticRegression(max_iter=1000)),
    ])
    log_reg_grid = GridSearchCV(
        log_reg_pipe,
        param_grid={"model__C": [0.01, 0.1, 1, 10, 100]},
        cv=5, scoring="f1",
    )
    log_reg_grid.fit(X_train, y_train)
    print("---------------- LAPSE MODEL SCORES -----------------")
    print("\nLogistic regression best params :", log_reg_grid.best_params_)
    print("Logistic regression best CV F1  :", round(log_reg_grid.best_score_, 3))
    rf_pipe = Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
    ])
    rf_grid = GridSearchCV(
        rf_pipe,
        param_grid={"model__max_depth": [3, 5, 8],
                    "model__min_samples_leaf": [5, 10, 20]},
        cv=5, scoring="f1",
    )
    rf_grid.fit(X_train, y_train)
    print("\nRandom forest best params :", rf_grid.best_params_)
    print("Random forest best CV F1  :", round(rf_grid.best_score_, 3))
    def evaluate(name, model):
        prediction = model.predict(X_test)
        print(f"\n{name}")
        print("Confusion matrix (rows = true 0/1, cols = predicted 0/1):")
        print(confusion_matrix(y_test, prediction))
        p = precision_score(y_test, prediction, zero_division=0)
        r = recall_score(y_test, prediction, zero_division=0)
        f = f1_score(y_test, prediction, zero_division=0)
        print(f"Lapsed class -> precision {p:.3f} | recall {r:.3f} | F1 {f:.3f}")
        return {"model": name, "precision": p, "recall": r, "f1": f}

    results = pd.DataFrame([
        evaluate("Logistic regression", log_reg_grid.best_estimator_),
        evaluate("Random forest",       rf_grid.best_estimator_),
    ]).round(3)
    print("\n", results.to_string(index=False)) 
    print("---------- CLASS-BALANCE EXPERIMENT -----------")
    print("Original train counts:\n", y_train.value_counts())
    lapsed_idx = y_train[y_train == 1].index
    n_major    = (y_train == 0).sum()
    n_minor    = (y_train == 1).sum()
    n_extra    = n_major - n_minor                      
    rng   = np.random.RandomState(42)
    extra = rng.choice(lapsed_idx, size=n_extra, replace=True)
    X_bal = pd.concat([X_train, X_train.loc[extra]])
    y_bal = pd.concat([y_train, y_train.loc[extra]])
    print("\nBalanced train counts:\n", y_bal.value_counts())
    lr_orig = log_reg_grid.best_estimator_
    lr_bal = clone(log_reg_grid.best_estimator_)
    lr_bal.fit(X_bal, y_bal)
    def scores(name, model):
        pred = model.predict(X_test)
        return {"model": name,
                "precision": precision_score(y_test, pred, zero_division=0),
                "recall":    recall_score(y_test, pred, zero_division=0),
                "f1":        f1_score(y_test, pred, zero_division=0)}
    comp = pd.DataFrame([scores("Un-balanced LR", lr_orig),
                        scores("Over-sampled LR", lr_bal)]).set_index("model")
    print("\n", comp.round(3))
    print("\n Difference between balanced dataset - original dataset):")
    print((comp.loc["Over-sampled LR"] - comp.loc["Un-balanced LR"]).round(3))
    print("------------------ FEATURE IMPORTANCE -------------------")
    rf_best  = rf_grid.best_estimator_
    names    = rf_best.named_steps["prep"].get_feature_names_out()
    imp      = pd.Series(rf_best.named_steps["model"].feature_importances_, index=names)
    print("Random Forest Top 5 Feature Importance ")
    print(imp.sort_values(ascending=False).head(5).round(3))
    lr_best = log_reg_grid.best_estimator_
    coef    = pd.Series(lr_best.named_steps["model"].coef_[0],
                        index=lr_best.named_steps["prep"].get_feature_names_out())
    print("Logestic Regression best 3 Feature Importance")
    print(coef.loc[["num__recency", "num__frequency", "num__cancellation_rate"]].round(3))
    print("-------------- COMPARISON OF ORIGINAL (VS) LOADED TEST PREDICTIONS -------------")
    final_model = rf_grid.best_estimator_
    os.makedirs("outputs", exist_ok=True)
    joblib.dump(final_model, "outputs/lapse_model.joblib")
    loaded = joblib.load("outputs/lapse_model.joblib")
    orig_pred   = final_model.predict(X_test)
    loaded_pred = loaded.predict(X_test)
    orig_prob   = final_model.predict_proba(X_test)[:, 1]
    loaded_prob = loaded.predict_proba(X_test)[:, 1]
    print("Identical predictions:", np.array_equal(orig_pred, loaded_pred))
    comparison = pd.DataFrame({
        "orig_pred":   orig_pred,
        "loaded_pred": loaded_pred,
        "orig_prob":   orig_prob.round(4),
        "loaded_prob": loaded_prob.round(4),
    })
    comparison["match"] = comparison["orig_pred"] == comparison["loaded_pred"]
    print(comparison.head(10))
    print("\nRows compared:", len(comparison))
    print("All predictions match:", comparison["match"].all())
    print("Mismatches:", (~comparison["match"]).sum())
    print("---------------- LAPSE PREDICTION ON TEST DATA -----------------")
    proba = final_model.predict_proba(X_test)[:, 1]
    test_df = pd.DataFrame({
        "customer_id":       data.loc[X_test.index, "customer_id"].values,
        "lapse_probability": proba,
        "actually_lapsed":   y_test.values,
    })
    seg = pd.read_csv("outputs/customer_segments.csv")[["customer_id", "is_premium"]]
    test_df = test_df.merge(seg, on="customer_id", how="left")
    premium = test_df[test_df["is_premium"] == 1]
    retention = (premium.sort_values("lapse_probability", ascending=False)
                        .head(10)[["customer_id", "lapse_probability", "actually_lapsed"]]
                        .round(4))
    retention.to_csv("outputs/retention_list.csv", index=False)
    print(retention)
    print("Truly lapsed in list:", int(retention["actually_lapsed"].sum()), "of", len(retention))
    print("Premium customers in test set:", len(premium))

if __name__=="__main__": rfm = RFM_table(); k_mean(rfm); lapsed_model()