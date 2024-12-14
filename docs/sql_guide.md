# SQL Guide for Data Analysis

## Introduction
SQL (Structured Query Language) is essential for data analysts to retrieve, manipulate, and analyze data in relational databases. This guide covers key concepts and common operations.

## Basic Queries

### SELECT Statement
```sql
-- Basic SELECT
SELECT column1, column2
FROM table_name;

-- Select all columns
SELECT *
FROM table_name;

-- Select distinct values
SELECT DISTINCT column
FROM table_name;
```

### Filtering Data
```sql
-- WHERE clause
SELECT *
FROM table_name
WHERE condition;

-- Multiple conditions
SELECT *
FROM table_name
WHERE condition1 AND condition2;

-- IN operator
SELECT *
FROM table_name
WHERE column IN (value1, value2);
```

### Sorting Data
```sql
-- ORDER BY
SELECT *
FROM table_name
ORDER BY column ASC;

-- Multiple columns
SELECT *
FROM table_name
ORDER BY column1 DESC, column2 ASC;
```

## Aggregations

### Basic Aggregations
```sql
-- COUNT
SELECT COUNT(*) as total_rows
FROM table_name;

-- SUM, AVG, MIN, MAX
SELECT 
    SUM(column) as total,
    AVG(column) as average,
    MIN(column) as minimum,
    MAX(column) as maximum
FROM table_name;
```

### GROUP BY
```sql
-- Basic grouping
SELECT category, COUNT(*) as count
FROM table_name
GROUP BY category;

-- Multiple columns
SELECT category, sub_category, SUM(value) as total
FROM table_name
GROUP BY category, sub_category;

-- HAVING clause
SELECT category, COUNT(*) as count
FROM table_name
GROUP BY category
HAVING COUNT(*) > 5;
```

## Joins

### Types of Joins
```sql
-- INNER JOIN
SELECT t1.column, t2.column
FROM table1 t1
INNER JOIN table2 t2
ON t1.key = t2.key;

-- LEFT JOIN
SELECT t1.column, t2.column
FROM table1 t1
LEFT JOIN table2 t2
ON t1.key = t2.key;

-- Multiple joins
SELECT t1.col, t2.col, t3.col
FROM table1 t1
LEFT JOIN table2 t2 ON t1.key = t2.key
LEFT JOIN table3 t3 ON t2.key = t3.key;
```

## Advanced Operations

### Subqueries
```sql
-- Subquery in WHERE
SELECT *
FROM table1
WHERE column IN (
    SELECT column
    FROM table2
    WHERE condition
);

-- Subquery in FROM
SELECT *
FROM (
    SELECT column, COUNT(*) as count
    FROM table_name
    GROUP BY column
) subquery
WHERE count > 5;
```

### Window Functions
```sql
-- ROW_NUMBER
SELECT *,
    ROW_NUMBER() OVER (
        PARTITION BY category
        ORDER BY value DESC
    ) as rank
FROM table_name;

-- Running totals
SELECT *,
    SUM(value) OVER (
        ORDER BY date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) as running_total
FROM table_name;
```

### Common Table Expressions (CTEs)
```sql
WITH cte_name AS (
    SELECT column, COUNT(*) as count
    FROM table_name
    GROUP BY column
)
SELECT *
FROM cte_name
WHERE count > 5;
```

## Best Practices

1. Always use meaningful table aliases
2. Write readable, formatted queries
3. Use appropriate indexing
4. Optimize JOIN operations
5. Avoid SELECT *
6. Use CTEs for complex queries
7. Comment your queries

## Performance Tips

1. Filter early in the query
2. Use appropriate indexes
3. Avoid correlated subqueries
4. Use EXPLAIN to analyze queries
5. Limit result sets appropriately

## Common Pitfalls

1. Not handling NULL values properly
2. Incorrect JOIN conditions
3. Forgetting WHERE clauses
4. Using inefficient wildcards
5. Not considering data types

## Data Analysis Examples

### Customer Analysis
```sql
-- Customer purchase patterns
SELECT 
    customer_id,
    COUNT(*) as total_orders,
    SUM(amount) as total_spent,
    AVG(amount) as avg_order_value
FROM orders
GROUP BY customer_id
ORDER BY total_spent DESC;
```

### Time Series Analysis
```sql
-- Monthly trends
SELECT 
    DATE_TRUNC('month', date) as month,
    COUNT(*) as total_transactions,
    SUM(amount) as total_revenue
FROM transactions
GROUP BY DATE_TRUNC('month', date)
ORDER BY month;
```

### Cohort Analysis
```sql
-- Customer cohorts
WITH first_purchase AS (
    SELECT 
        customer_id,
        MIN(DATE_TRUNC('month', date)) as cohort_month
    FROM orders
    GROUP BY customer_id
)
SELECT 
    cohort_month,
    COUNT(DISTINCT customer_id) as cohort_size
FROM first_purchase
GROUP BY cohort_month
ORDER BY cohort_month;
```

## Resources

- [SQL Style Guide](https://www.sqlstyle.guide/)
- [Mode Analytics SQL Tutorial](https://mode.com/sql-tutorial/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/) 