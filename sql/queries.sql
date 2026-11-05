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