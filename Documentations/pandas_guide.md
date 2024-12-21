# Pandas Guide for Data Analysis

## Introduction
Pandas is a powerful Python library for data manipulation and analysis. This guide covers essential concepts and operations that every data analyst should know.

## Basic Operations

### Reading Data
```python
# Read CSV file
df = pd.read_csv('data.csv')

# Read Excel file
df = pd.read_excel('data.xlsx')

# Read JSON file
df = pd.read_json('data.json')
```

### Data Inspection
```python
# View first few rows
df.head()

# View basic information
df.info()

# Get summary statistics
df.describe()
```

### Data Selection
```python
# Select single column
df['column_name']

# Select multiple columns
df[['col1', 'col2']]

# Filter rows
df[df['column'] > 5]
```

## Data Cleaning

### Handling Missing Values
```python
# Check for missing values
df.isnull().sum()

# Drop rows with missing values
df.dropna()

# Fill missing values
df.fillna(value)
```

### Removing Duplicates
```python
# Check for duplicates
df.duplicated().sum()

# Remove duplicates
df.drop_duplicates()
```

## Data Transformation

### Column Operations
```python
# Create new column
df['new_col'] = df['col1'] + df['col2']

# Apply function to column
df['col'].apply(function)

# Rename columns
df.rename(columns={'old_name': 'new_name'})
```

### Grouping and Aggregation
```python
# Group by single column
df.groupby('category').mean()

# Group by multiple columns
df.groupby(['cat1', 'cat2']).sum()

# Multiple aggregations
df.groupby('category').agg({
    'col1': 'mean',
    'col2': 'sum',
    'col3': 'count'
})
```

## Advanced Operations

### Merging Data
```python
# Merge DataFrames
pd.merge(df1, df2, on='key', how='left')

# Concatenate DataFrames
pd.concat([df1, df2])
```

### Pivot Tables
```python
# Create pivot table
pd.pivot_table(
    df,
    values='value',
    index='row_categories',
    columns='column_categories',
    aggfunc='mean'
)
```

### Time Series Operations
```python
# Set datetime index
df.set_index('date', inplace=True)

# Resample time series
df.resample('M').mean()

# Rolling calculations
df.rolling(window=7).mean()
```

## Best Practices

1. Always check data types and missing values first
2. Make copies of DataFrames before modifications
3. Use method chaining for cleaner code
4. Document your data transformations
5. Handle missing values appropriately
6. Use vectorized operations instead of loops

## Common Pitfalls

1. Modifying views instead of copies
2. Ignoring data types in calculations
3. Not handling missing values properly
4. Using inefficient operations in loops
5. Forgetting to handle duplicates

## Performance Tips

1. Use appropriate data types
2. Avoid unnecessary copies
3. Use vectorized operations
4. Filter data early in the process
5. Use appropriate indexing

## Resources

- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Pandas Cheat Sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
- [10 Minutes to Pandas](https://pandas.pydata.org/docs/user_guide/10min.html) 