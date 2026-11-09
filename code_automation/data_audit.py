import pandas as pd
import numpy as np
import os
import sqlite3

customers_df = pd.read_csv("data/raw/customers.csv")
orders_df = pd.read_csv("data/raw/orders.csv")
products_df = pd.read_csv("data/raw/products.csv")
tickets_df = pd.read_csv(
    "data/raw/tickets.csv", keep_default_na=False
) 

def audit():
    print(f"\n{'-'*25}\n--- ORDERS AUDIT ---\n{'-'*25}")
    print("Number of rows in ORDERS dataset", orders_df.shape)
    total_orders = len(orders_df)
    exact_dupes_orders = orders_df.duplicated().sum()
    pending_rows = (orders_df["status"].str.lower() == "pending").sum() 
    distinct_order_ids = (orders_df["order_id"].nunique())
    non_numeric_amounts = pd.to_numeric(orders_df["order_amount"], errors="coerce").isna().sum()
    deliv_numeric = pd.to_numeric(orders_df["delivery_minutes"], errors="coerce")
    neg_delivery = (deliv_numeric < 0).sum()
    blank_delivery = (
            orders_df["delivery_minutes"].isna()
            | (orders_df["delivery_minutes"] == "")
            | (orders_df["delivery_minutes"].astype(str).str.strip() == "")
        ).sum()
    print(f"Number of exact duplicate rows in ORDERS dataset: {exact_dupes_orders}")
    print(f"Total number of pending rows in STATUS columns: {pending_rows}")
    print(f"Number of distinct order IDs in ORDERS dataset: {distinct_order_ids}")
    print(f"Number of non-numeric ORDER AMOUNT columns: {non_numeric_amounts}")
    print(f"Number of negative values in DELIVERY MINUTES column: {neg_delivery}")
    print(f"Number of blank values in DELIVERY MINUTES column: {blank_delivery}")
    print(f"\n{'-'*25}\n--- CUSTOMER AUDIT---\n{'-'*25}")
    age_numeric = pd.to_numeric(customers_df["age"], errors="coerce")
    blank_age = (
            customers_df["age"].isna()
            | (customers_df["age"] == "")
            | (customers_df["age"].astype(str).str.strip() == "")
        ).sum()
    age_outside = ((age_numeric < 10) | (age_numeric > 100)).sum()
    distinct_cities = customers_df["city"].nunique()
    blank_gender = (
            customers_df["gender"].isna()
            | (customers_df["gender"] == "")
            | (customers_df["gender"].astype(str).str.strip().str.lower() == "")
        ).sum()
    distinct_genders = (customers_df["gender"].nunique())
    print(f"Number of blank values in AGE column: {blank_age}")
    print(f"Number of values outside the 10-100 range in AGE column: {age_outside}")
    print(f"Number of unique cities in CITY column: {distinct_cities}")
    print(f"Number of unique raw genders in GENDER column: {distinct_genders}")
    print(f"Number of blank values in GENDER column: {blank_gender}")

def data_clean():
    orders_df = pd.read_csv("data/raw/orders.csv")
    customers_df = pd.read_csv("data/raw/customers.csv")
    products_df = pd.read_csv("data/raw/products.csv")
    print(f"\n{'-'*30}\n--- ORDERS DATASET CLEAN ---\n{'-'*30}")
    orders_df["order_ts"] = pd.to_datetime(orders_df["order_ts"], errors = "coerce")
    orders_df = orders_df.drop_duplicates()
    orders_df = (
        orders_df.sort_values(by="order_ts", ascending=True)
        .drop_duplicates(subset=["order_id"], keep="last")
        .reset_index(drop=True) )
    orders_df["order_amount"] = pd.to_numeric(orders_df["order_amount"], errors="coerce")
    merged_orders = orders_df.merge(
        products_df[["product_id", "price_inr"]], 
        on="product_id", 
        how="left", 
        suffixes=("", "_prod")
    )
    calculated_amount = round(
        merged_orders["price_inr"]
        * merged_orders["quantity"]
        * (1 - merged_orders["discount_pct"] / 100),
        2,
    )
    orders_df["order_amount"] = orders_df["order_amount"].fillna(calculated_amount)
    orders_df["delivery_minutes"] = pd.to_numeric(
        orders_df["delivery_minutes"], errors="coerce")
    orders_df.loc[orders_df["delivery_minutes"] < 0, "delivery_minutes"] = np.nan
    completed_mask = (orders_df["status"].astype(str).str.strip().str.lower() == "completed")
    cancelled_mask = (orders_df["status"].astype(str).str.strip().str.lower() == "cancelled")
    category_medians = (
        orders_df[completed_mask]
        .groupby("product_id")["delivery_minutes"]
        .transform("median")
    )
    orders_df.loc[completed_mask, "delivery_minutes"] = orders_df.loc[
        completed_mask, "delivery_minutes"
    ].fillna(category_medians)
    orders_df["order_amount"] = pd.to_numeric(
        orders_df["order_amount"], errors="coerce"
    )
    sum_of_completed_orders = orders_df.loc[completed_mask, "order_amount"].sum()
    print(f"Cleaned rows: {len(orders_df)}")
    print(f"Total number of COMPLETED rows:", completed_mask.sum())
    print(f"Total number of CANCELLED rows:", cancelled_mask.sum())
    print("Remaining missing ORDER_AMOUNT after impute:",orders_df["order_amount"].isna().sum())
    print("Remaining missing DELIVERY_MINUTES for completed orders:",orders_df[completed_mask]["delivery_minutes"].isna().sum())
    print(f"Sum of order amount for completed orders: {sum_of_completed_orders}")
    print(f"\n{'-'*40}\n--- CUSTOMER DATASET CLEAN ---\n{'-'*40}")
    customers_df["age"] = pd.to_numeric(customers_df["age"], errors="coerce")
    customers_df.loc[(customers_df["age"] < 10) | (customers_df["age"] > 100), "age"] = (np.nan)
    customers_df["age"] = customers_df["age"].fillna(
        customers_df.groupby("city")["age"].transform("median"))
    customers_df["gender"] = customers_df["gender"].replace(r"^\s*$", np.nan, regex=True)
    customers_df["gender"] = (customers_df["gender"].fillna("unknown").astype(str)
        .str.strip().str.lower())
    gender_mapping = {
        "male": "male","m": "male",
        "female": "female","f": "female",
        "unknown": "unknown",
    }
    customers_df["city"] = customers_df["city"].str.lower().str.strip()
    city_mapping = {
        'pune': 'Pune','delhi': 'Delhi','bengaluru': 'Bengaluru',
        'chennai': 'Chennai','hyderabad' : 'Hyderabad','mumbai' : 'Mumbai',
    }
    customers_df['city'] = customers_df['city'].map(city_mapping)
    customers_df["signup_date"] = pd.to_datetime(customers_df["signup_date"], errors='coerce')
    customers_df["gender"] = customers_df["gender"].map(gender_mapping).fillna("unknown")
    missing = customers_df['age'].isna().sum()
    out_of_range = (~customers_df['age'].between(18, 60) & customers_df['age'].notna()).sum()
    print("Total count in AGE column after fillna", customers_df["age"].count())
    print(f"Total count of each gender category after cleaning:")
    print(customers_df["gender"].value_counts())
    print(f"Total count of each city category after cleaning:")
    print(customers_df["city"].value_counts())
    print("Number of Missing ages:", missing)
    print("Number of Out of range ages between 18 and 60:", out_of_range)

def validate():
    os.makedirs("sql", exist_ok=True)
    queries_content = """
    -- Q1: WHERE then HAVING (Completed orders > 500, grouped by customer, total > 8000)
    SELECT customer_id, SUM(order_amount) AS total_spent
    FROM orders
    WHERE LOWER(status) = 'completed' AND order_amount > 500
    GROUP BY customer_id
    HAVING SUM(order_amount) > 8000;

    -- Q2: Three-table join (Revenue by city and product category)
    SELECT 
        c.city, 
        p.category AS product_category, 
        SUM(o.order_amount) AS total_revenue
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    JOIN products p ON o.product_id = p.product_id
    WHERE LOWER(o.status) = 'completed'
    GROUP BY c.city, p.category
    ORDER BY total_revenue DESC;

    -- Q3: Scalar subquery (Products priced above the average catalogue price)
    SELECT product_id, product_name, price_inr
    FROM products
    WHERE price_inr > (SELECT AVG(price_inr) FROM products);

    -- Q4: LEFT JOIN (Customers who have no completed order)
    SELECT c.customer_id
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id AND LOWER(o.status) = 'completed'
    WHERE o.order_id IS NULL;

    -- Q5: Common Table Expressions (Customers whose total completed spend is above the overall average)
    WITH customer_spend AS (
        SELECT customer_id, SUM(order_amount) AS total_spend
        FROM orders
        WHERE LOWER(status) = 'completed'
        GROUP BY customer_id
    ),
    avg_spend AS (
        SELECT AVG(total_spend) AS mean_spend FROM customer_spend
    )
    SELECT cs.customer_id, cs.total_spend
    FROM customer_spend cs, avg_spend av
    WHERE cs.total_spend > av.mean_spend;

    -- Q6: Ranking Window Functions (Rank, Dense Rank, Row Number partitioned by city)
    WITH customer_city_spend AS (
        SELECT 
            c.customer_id, 
            c.city, 
            SUM(o.order_amount) AS total_spend
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        WHERE LOWER(o.status) = 'completed'
        GROUP BY c.customer_id, c.city
    )
    SELECT 
        city,
        customer_id,
        total_spend,
        RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS rnk,
        DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS dense_rnk,
        ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) AS row_num
    FROM customer_city_spend;

    -- Q7: Running Total (Daily completed revenue and cumulative sum)
    WITH daily_revenue AS (
        SELECT 
            SUBSTR(order_ts, 1, 10) AS day,
            SUM(order_amount) AS daily_rev
        FROM orders
        WHERE LOWER(status) = 'completed'
        GROUP BY SUBSTR(order_ts, 1, 10)
    )
    SELECT 
        day,
        daily_rev,
        SUM(daily_rev) OVER (ORDER BY day) AS cumulative_revenue
    FROM daily_revenue
    ORDER BY day;
    """
    with open("sql/queries.sql", "w") as f:
        f.write(queries_content.strip())
    conn = sqlite3.connect("data/instamart.db")

    with open("sql/queries.sql", "r") as f:
        sql_script = f.read()

    queries = [q.strip() for q in sql_script.split(";") if q.strip()]

    for i, query in enumerate(queries, 1):
        print("=" * 60)
        print(f"RUNNING QUERY Q{i}")
        print("=" * 60)

        result_df = pd.read_sql_query(query, conn)

        if i == 1:
            print(f"Result Checkpoint -> Total Customers: {len(result_df)}")

        elif i == 2:
            print(f"Result Checkpoint -> Total Rows: {len(result_df)}")
            print(f"Total Revenue Sum: INR {result_df['total_revenue'].sum():,.2f}")

        elif i == 3:
            print(f"Result Checkpoint -> Total Products: {len(result_df)}")

        elif i == 4:
            print(f"Result Checkpoint -> Total Customers: {len(result_df)}")

        elif i == 5:
            print(f"Result Checkpoint -> Total Customers: {len(result_df)}")
            # Average of ALL customers' spend (not just those above average).
            avg_df = pd.read_sql_query("""
                SELECT AVG(total_spend) AS mean_spend FROM (
                    SELECT customer_id, SUM(order_amount) AS total_spend
                    FROM orders
                    WHERE LOWER(status) = 'completed'
                    GROUP BY customer_id
                )
            """, conn)
            print(f"Average Spend: INR {avg_df['mean_spend'].iloc[0]:,.2f}")

        elif i == 6:
            print(f"Result Checkpoint -> Total Rows: {len(result_df)}")

            print("\nTop customer per city (row_num = 1):")
            top = result_df[result_df["row_num"] == 1][["city", "customer_id", "total_spend"]]
            print(top.sort_values("city").to_string(index=False))

            diff = (result_df["rnk"] != result_df["row_num"]).sum()
            print(f"\nRows where RANK() != ROW_NUMBER(): {diff}")
            print("Matches expected 8?", diff == 8)

            print("\nRows affected by ties:")
            print(result_df[result_df["rnk"] != result_df["row_num"]].to_string(index=False))

        elif i == 7:
            print(f"Result Checkpoint -> Total Daily Rows: {len(result_df)}")
            last_cum_val = result_df["cumulative_revenue"].iloc[-1]
            print(f"Last Cumulative Revenue: INR {last_cum_val:,.2f}")

        print(result_df.head())
        print("\n")
    print("Successfully created sql/queries.sql!\n")

def cross_check():
    orders_df = pd.read_csv("data/clean/orders.csv")
    customers_df = pd.read_csv("data/clean/customers.csv")
    products_df = pd.read_csv("data/clean/products.csv")
    orders_df["order_amount"] = pd.to_numeric(orders_df["order_amount"], errors="coerce")
    cat_col = "category"
    full = (orders_df
            .merge(customers_df[["customer_id", "city"]], on="customer_id")
            .merge(products_df[["product_id", cat_col]], on="product_id"))
    df = full[full["status"].str.strip().str.lower() == "completed"]
    pt = pd.pivot_table(
        df,
        index="city",
        columns=cat_col,
        values="order_amount",
        aggfunc="sum",
        margins=True,
        margins_name="Total",
        fill_value=0
    )
    pandas_total = pt.loc["Total", "Total"]
    city_stats = (df.groupby("city")
                    .agg(total_amount=("order_amount", "sum"),
                         mean_amount=("order_amount", "mean"),
                         order_count=("order_amount", "count"))
                    .round(2))
    print(city_stats)
    print("Pandas grand total: INR", round(pandas_total, 2))
    import sqlite3
    con = sqlite3.connect("data/instamart.db")
    sql_total = pd.read_sql_query(
        "SELECT SUM(order_amount) AS t FROM orders WHERE status = 'completed'", con
    )["t"].iloc[0]
    con.close()

    print("SQL total: INR", round(sql_total, 2))
    print("Match:", np.isclose(pandas_total, sql_total, atol=0.01))

def exploratory_analysis():
    df_orders    = pd.read_csv("data/clean/orders.csv")
    df_customers = pd.read_csv("data/clean/customers.csv")
    df_products  = pd.read_csv("data/clean/products.csv")

    cat_col = "category"     # <-- your category column name in products
    full = (df_orders
            .merge(df_customers[["customer_id", "city"]], on="customer_id")
            .merge(df_products[["product_id", cat_col]].drop_duplicates("product_id"),
                on="product_id"))
    done_b = full[full["status"].str.strip().str.lower() == "completed"].copy()
    x = done_b["order_amount"]

    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr    = q3 - q1
    lower  = q1 - 1.5 * iqr
    upper  = q3 + 1.5 * iqr

    stats = pd.Series({
        "Mean": x.mean(),
        "Median": x.median(),
        "Skewness": x.skew(),
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "Lower fence": lower,
        "Upper fence": upper,
        "Outliers below lower fence": int((x < lower).sum()),
        "Outliers above upper fence": int((x > upper).sum()),
    }).round(2)
    print("Rows:", len(done_b), "| Total:", round(done_b["order_amount"].sum(), 2))
    print(stats)
    corr = done_b[["order_amount", "discount_pct", "quantity", "delivery_minutes"]].corr()
    print(corr.round(3))
    pairs = corr.abs().where(~np.eye(len(corr), dtype=bool)).stack()
    a, b = pairs.idxmax()
    print("Largest |corr|:", a, "&", b, "=", round(corr.loc[a, b], 3))

if __name__=="__main__": audit(); data_clean(); validate(); cross_check(); exploratory_analysis()